# -*- coding: utf-8 -*-

"""
The script runs a "classic" NER for two types of entities — CHEMICAL and SPECIES — on texts from articles, listed in Excel (Paper column).
It loads pre-trained biomedical models from spaCy,
extracts entities from the column with agent responses,
compiles the results (including coordinates and the context sentence),
prints summary statistics, and
saves the results in a CSV file.
"""



import os, re, json, time, requests
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm




#  SETTINGS

INPUT_XLSX = "../data_input/Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx"
SHEET_NAME = "Agent eval results"
ID_COL     = "KER"
PAPER_COL  = "Paper"
MIN_LEN    = 3
OUT_CSV    = "../data_output/entities_ner"
NCBI_EMAIL = "grrralena@outlook.de"  
NCBI_API_KEY = "04e0fad15051238d79f1898154016f70d508"  

CACHE_DIR = Path("./_paper_cache")
if CACHE_DIR.exists() and not CACHE_DIR.is_dir():
    raise RuntimeError(f"'{CACHE_DIR}' существует, но это файл — удалите его.")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHEM_MODEL = "en_ner_bc5cdr_md"
SPEC_MODEL = "en_ner_bionlp13cg_md"



# === HELPERS ===

#Cleans up the string by replacing multiple spaces with a single space and trimming the edges. 
#If the input isn't a string, it returns an empty string. This is necessary for consistent processing of texts and entity names.


def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def extract_text_from_pdf(pdf_path: Path) -> str:
    import fitz  # pymupdf
    txt = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt.append(page.get_text())
    return sanitize_text(" ".join(txt))

PMID_RE = re.compile(r"PMID[_\- ]?(\d+)", re.IGNORECASE)
def extract_pmid(s: str) -> str | None:
    m = PMID_RE.search(str(s))
    return m.group(1) if m else None

def cache_pdf_path(pmid): return CACHE_DIR / f"{pmid}.pdf"
def cache_txt_path(pmid): return CACHE_DIR / f"{pmid}.txt"

def europe_pmc_pdf_url(pmid: str) -> str | None:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return None
    res = results[0]
    ft_list = res.get("fullTextUrlList", {}).get("fullTextUrl", [])
    for item in ft_list:
        if item.get("documentStyle") == "pdf" and "url" in item:
            return item["url"]
    pmcid = res.get("pmcid")
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    return None

def pubmed_title_abstract(pmid: str) -> str | None:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {"db": "pubmed", "id": pmid, "retmode": "text", "rettype": "abstract"}
    r = requests.get(f"{base}/efetch.fcgi", params=params, timeout=10)
    if r.status_code != 200:
        return None
    return sanitize_text(r.text)

def resolve_text_for_paper_cell(paper_cell: str) -> tuple[str, str]:
    pmid = extract_pmid(paper_cell)
    if not pmid:
        return "", ""

    pdf_path = cache_pdf_path(pmid)
    if pdf_path.exists():
        return "pdf", extract_text_from_pdf(pdf_path)

    try:
        pdf_url = europe_pmc_pdf_url(pmid)
        if pdf_url:
            r = requests.get(pdf_url, timeout=15)
            if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                pdf_path.write_bytes(r.content)
                return "pdf", extract_text_from_pdf(pdf_path)
    except requests.exceptions.Timeout:
        print(f"[WARN] Timeout for {pmid}, fallback to abstract")
    except Exception as e:
        print(f"[WARN] PDF error for {pmid}: {e}")

    abs_txt = pubmed_title_abstract(pmid)
    if abs_txt:
        cache_txt_path(pmid).write_text(abs_txt, encoding="utf-8")
        return "abstract", abs_txt

    return "", ""



# Takes a list of highlighted segments of entities [(start, end, text, type), ...], sorts them, and merges overlapping spans of the same type to prevent duplicates or overlaps. 
# Returns a "stitched" list of spans.


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


#Loads two pre-trained spaCy models:
#en_ner_bc5cdr_md for chemical entities,
#en_ner_bionlp13cg_md for species (Species/Organism).
#Disables unnecessary components (tagger, lemmatizer, textcat) and, if possible, adds simple sentence segmentation (sentencizer).
# Returns a pair of pipelines (nlp_chem, nlp_spec)


def load_models():
    print("[i] Loading spaCy models...")
    nlp_chem = spacy.load(CHEM_MODEL, disable=["tagger","lemmatizer","textcat"])
    nlp_spec = spacy.load(SPEC_MODEL, disable=["tagger","lemmatizer","textcat"])
    # Очищаем вектора для экономии памяти
    nlp_chem.vocab.vectors.clear()
    nlp_spec.vocab.vectors.clear()
    for nlp in (nlp_chem, nlp_spec):
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    print("[i] Models loaded successfully")
    return nlp_chem, nlp_spec


# Main extraction function:

### extract_entities
###
# Purpose:
#   To extract CHEMICAL and SPECIES/ORGANISM entities from text using two
#   pre-loaded spaCy models and return a list of dictionaries containing:
#   entity, type, start, end, sentence.
#
# Expected dependencies:
#   - functions: sanitize_text(str) -> str, merge_overlaps(list[tuple]) -> list[tuple]
#   - constant: MIN_LEN (minimum length of the entity text)
#   - nlp_chem: spaCy Language model for chemical entities
#   - nlp_spec: spaCy Language model for species/organisms (with sentence segmentation setup)
#
# Return value:
#   list[dict]: [{"entity": str, "type": "CHEMICAL"|"SPECIES", "start": int, "end": int, "sentence": str}, ...]



def extract_entities_batch(texts: list[str], nlp_chem, nlp_spec, min_len=3):
    """Batched NER — принимает список текстов, возвращает список списков сущностей"""
    chem_docs = list(nlp_chem.pipe(texts, batch_size=4, n_process=1))
    spec_docs = list(nlp_spec.pipe(texts, batch_size=4, n_process=1))

    all_results = []
    for doc_chem, doc_spec, text in zip(chem_docs, spec_docs, texts):
        chem_spans = [(e.start_char, e.end_char, e.text, "CHEMICAL")
                      for e in doc_chem.ents if e.label_.upper() == "CHEMICAL"]
        species_spans = [(e.start_char, e.end_char, e.text, "SPECIES")
                         for e in doc_spec.ents if e.label_.upper() in {"SPECIES", "ORGANISM"}]
        chem_spans = merge_overlaps(chem_spans)
        species_spans = merge_overlaps(species_spans)
        sents = list(doc_spec.sents)

        def ctx(s, e):
            for se in sents:
                if se.start_char <= s and e <= se.end_char:
                    return se.text.strip()
            return text[max(0, s-120):min(len(text), e+120)].strip()

        ents_out = []
        for s, e, t, typ in chem_spans + species_spans:
            t = sanitize_text(t)
            if len(t) < min_len:
                continue
            ents_out.append({
                "type": typ,
                "entity": t,
                "char_start": s,
                "char_end": e,
                "context_sentence": ctx(s, e)
            })
        all_results.append(ents_out)
    return all_results

#   main execution - run code


df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
if ID_COL not in df.columns:
    df[ID_COL] = range(1, len(df) + 1)

nlp_chem, nlp_spec = load_models()
print(f"[i] Loaded {len(df)} rows from {SHEET_NAME}")

# Сначала собираем тексты
texts, refs, ids, modes = [], [], [], []
n_pdf = n_abs = n_fail = 0

for _, r in tqdm(df.iterrows(), total=len(df), desc="Resolving papers", unit="row"):
    paper_ref = r.get(PAPER_COL, "")
    mode, text = resolve_text_for_paper_cell(paper_ref)
    if not text:
        n_fail += 1
        continue
    if mode == "pdf":
        n_pdf += 1
    elif mode == "abstract":
        n_abs += 1
    texts.append(text)
    refs.append(paper_ref)
    ids.append(r.get(ID_COL))
    modes.append(mode)

print(f"[i] Texts resolved: {len(texts)}, PDF={n_pdf}, Abstract={n_abs}, Fail={n_fail}")

# Теперь прогоняем NER батчами
all_entities = extract_entities_batch(texts, nlp_chem, nlp_spec, MIN_LEN)

rows = []
for id_, ref, mode, ents in zip(ids, refs, modes, all_entities):
    for e in ents:
        rows.append({
            ID_COL: id_,
            "Paper": ref,
            "source": mode,
            "type": e["type"],
            "entity": e["entity"],
            "char_start": e["char_start"],
            "char_end": e["char_end"],
            "context_sentence": e["context_sentence"]
        })

out_df = pd.DataFrame(rows).drop_duplicates()
out_df.to_csv(OUT_CSV, index=False)
print(f"[OK] Saved {len(out_df)} entities to {OUT_CSV}")
print(f"[i] Stats: PDF={n_pdf}, Abstract={n_abs}, Failed={n_fail}")


