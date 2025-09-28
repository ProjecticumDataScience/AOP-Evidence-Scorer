"""
Классический NER (CHEMICAL, SPECIES) для Spyder.
Excel-файл должен лежать в той же папке, что и этот скрипт.
Результат сохраняется в CSV/Parquet в этой же папке.
"""

import os
import re
import pandas as pd
import spacy


# ==== 1. НАСТРОЙКИ ====
INPUT_XLSX = "../data_input/Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx"
SHEET_NAME = "Agent eval results"   # лист в Excel
ID_COL = "Paper"                    # используем колонку Paper как ID
TEXT_COL = "Agent answer"      # колонка с текстом для анализа
OUT_PREFIX = "../data_output/entities_ner"
MIN_LEN = 3
CHEM_MODEL = "en_ner_bc5cdr_md"
SPEC_MODEL = "en_ner_bionlp13cg_md"



# ==== 2. ФУНКЦИИ ====
def load_models():
    nlp_chem = spacy.load(CHEM_MODEL, disable=["tagger", "lemmatizer", "textcat"])
    nlp_spec = spacy.load(SPEC_MODEL, disable=["tagger", "lemmatizer", "textcat"])
    for nlp in (nlp_chem, nlp_spec):
        if "senter" not in nlp.pipe_names:
            try:
                nlp.add_pipe("sentencizer")
            except Exception:
                pass
    return nlp_chem, nlp_spec

def sanitize_text(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def compose_text(row):
    return sanitize_text(row.get(TEXT_COL, ""))

def merge_overlaps(spans):
    if not spans:
        return spans
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    merged = []
    cur_s, cur_e, cur_text, cur_type = spans[0]
    for s, e, txt, typ in spans[1:]:
        if typ == cur_type and s <= cur_e:
            if e > cur_e:
                cur_e = e
                cur_text = txt
        else:
            merged.append((cur_s, cur_e, cur_text, cur_type))
            cur_s, cur_e, cur_text, cur_type = s, e, txt, typ
    merged.append((cur_s, cur_e, cur_text, cur_type))
    return merged

def extract_entities(text, nlp_chem, nlp_spec):
    if not text:
        return []
    doc_chem = nlp_chem(text)
    doc_spec = nlp_spec(text)
    chem_spans = [(e.start_char, e.end_char, e.text, "CHEMICAL")
                  for e in doc_chem.ents if e.label_.upper() == "CHEMICAL"]
    spec_spans = [(e.start_char, e.end_char, e.text, "SPECIES")
                  for e in doc_spec.ents if e.label_.upper() in ("SPECIES", "ORGANISM")]
    chem_spans = merge_overlaps(chem_spans)
    spec_spans = merge_overlaps(spec_spans)
    spans = chem_spans + spec_spans
    sentences = list(doc_spec.sents)
    results = []
    for s, e, txt, typ in spans:
        txt_clean = sanitize_text(txt)
        if len(txt_clean) < MIN_LEN:
            continue
        sent_text = next((sent.text for sent in sentences if sent.start_char <= s <= sent.end_char), "")
        results.append({"entity": txt_clean, "type": typ, "start": s, "end": e, "sentence": sent_text})
    return results


# ==== 3. ОСНОВНОЙ КОД ====


# ==== 3. ОСНОВНОЙ КОД ====

# 1) Загружаем Excel
if not os.path.exists(INPUT_XLSX):
    raise FileNotFoundError(f"Excel-файл не найден: {INPUT_XLSX}")

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

# если нет идентификатора — создадим
if ID_COL not in df.columns:
    df[ID_COL] = range(1, len(df) + 1)

print(f"[i] Загружено {len(df)} строк из листа '{SHEET_NAME}'")
print(f"[i] Доступные колонки: {list(df.columns)}")

# берём нужный текстовый столбец
print(f"[i] Используем колонку: {TEXT_COL}")
print(df[[ID_COL, TEXT_COL]].head(3))

# 2) Загружаем модели
nlp_chem, nlp_spec = load_models()

# 3) Извлекаем сущности по всем строкам
results = []
for _, row in df.iterrows():
    text = compose_text(row)   # берёт текст из TEXT_COL ("Agent answer")
    ents = extract_entities(text, nlp_chem, nlp_spec)
    for e in ents:
        results.append({
            "Paper": row[ID_COL],
            "entity": e["entity"],
            "type": e["type"],              # CHEMICAL или SPECIES
            "char_start": e["start"],
            "char_end": e["end"],
            "context_sentence": e["sentence"]
        })

# 4) Собираем итоговый DataFrame
out_df = pd.DataFrame(results).drop_duplicates()

# 5) Быстрая сводка по типам и покрытию
if out_df.empty:
    print("\n[!] Сущности не найдены.")
else:
    type_counts = out_df["type"].value_counts().sort_index()
    print("\n=== Кол-во сущностей по типам ===")
    for t, c in type_counts.items():
        print(f"{t:9s}: {c}")

    covered = out_df.groupby("type")["Paper"].nunique().sort_index()
    print("\n=== Кол-во уникальных статей с упоминаниями ===")
    for t, n in covered.items():
        print(f"{t:9s}: {n} статей")

    total_rows = len(df)
    rows_with_any = out_df["Paper"].nunique()
    print(f"\nCoverage: {rows_with_any}/{total_rows} статей ({rows_with_any/total_rows:.1%}) содержат ≥1 сущность")

    print("\n=== Топ-10 CHEMICAL ===")
    top_chem = (out_df[out_df["type"]=="CHEMICAL"]["entity"]
                .str.strip().str.lower().value_counts().head(10))
    for ent, cnt in top_chem.items():
        print(f"{cnt:5d}  {ent}")

    print("\n=== Топ-10 SPECIES ===")
    top_species = (out_df[out_df["type"]=="SPECIES"]["entity"]
                   .str.strip().str.lower().value_counts().head(10))
    for ent, cnt in top_species.items():
        print(f"{cnt:5d}  {ent}")

# 6) Сохранение результатов
out_csv = f"{OUT_PREFIX}.csv"
out_parquet = f"{OUT_PREFIX}.parquet"
out_df.to_csv(out_csv, index=False)
try:
    out_df.to_parquet(out_parquet, index=False)
except Exception:
    pass

print(f"\n[OK] Извлечение завершено: {len(out_df)} сущностей")
print(f"CSV: {os.path.abspath(out_csv)}")
if os.path.exists(out_parquet):
    print(f"Parquet: {os.path.abspath(out_parquet)}")

# 7) Первые строки для быстрой проверки в Spyder
out_df.head(10)
