"""
The script runs a "classic" NER for two types of entities — CHEMICAL and SPECIES — on texts from Excel.
It loads pre-trained biomedical models from spaCy,
extracts entities from the column with agent responses,
compiles the results (including coordinates and the context sentence),
prints summary statistics, and
saves the results in a CSV file.
"""

import os
import re
import pandas as pd
import spacy


# SETTINGS 
INPUT_XLSX = "../data_input/Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx"
SHEET_NAME = "Agent eval results"   # лист в Excel
ID_COL = "Paper"                    # используем колонку Paper как ID
TEXT_COL = "Agent answer"      # колонка с текстом для анализа
OUT_PREFIX = "../data_output/entities_ner"
MIN_LEN = 3
CHEM_MODEL = "en_ner_bc5cdr_md"
SPEC_MODEL = "en_ner_bionlp13cg_md"



# FUNCTIONS

#Loads two pre-trained spaCy models:
#en_ner_bc5cdr_md for chemical entities,
#en_ner_bionlp13cg_md for species (Species/Organism).
#Disables unnecessary components (tagger, lemmatizer, textcat) and, if possible, adds simple sentence segmentation (sentencizer).
# Returns a pair of pipelines (nlp_chem, nlp_spec)



def load_models():
    """Load pre-trained spaCy models for chemicals and species.

    Models:
        - en_ner_bc5cdr_md for chemical entities
        - en_ner_bionlp13cg_md for species/organisms

    Notes:
        Disables tagger, lemmatizer, textcat. Adds a sentencizer if missing.

    Returns:
        Tuple[Language, Language]: (nlp_chem, nlp_spec)
    """
    
    nlp_chem = spacy.load(CHEM_MODEL, disable=["tagger", "lemmatizer", "textcat"])
    nlp_spec = spacy.load(SPEC_MODEL, disable=["tagger", "lemmatizer", "textcat"])
    for nlp in (nlp_chem, nlp_spec):
        if "senter" not in nlp.pipe_names:
            try:
                nlp.add_pipe("sentencizer")
            except Exception:
                pass
    return nlp_chem, nlp_spec


#Cleans up the string by replacing multiple spaces with a single space and trimming the edges. 
#If the input isn't a string, it returns an empty string. This is necessary for consistent processing of texts and entity names.

def sanitize_text(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


#It takes the text from the TEXT_COL column of the dataframe (default is "Agent answer") and runs it through sanitize_text.
#  This is the single point where it decides which text to analyze.

def compose_text(row):
    return sanitize_text(row.get(TEXT_COL, ""))

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



def extract_entities(text, nlp_chem, nlp_spec):
    ### quick check for empty input ###
    if not text:
        return []

    ### running the text through two models (chemistry and species)  ###
    doc_chem = nlp_chem(text)
    doc_spec = nlp_spec(text)

    ### gathering raw spans from each model   ###
    # CHEMICAL from the chemical model
    chem_spans = [
        (e.start_char, e.end_char, e.text, "CHEMICAL")
        for e in doc_chem.ents
        if e.label_.upper() == "CHEMICAL"
    ]
    # SPECIES/ORGANISM from the species model
    spec_spans = [
        (e.start_char, e.end_char, e.text, "SPECIES")
        for e in doc_spec.ents
        if e.label_.upper() in ("SPECIES", "ORGANISM")
    ]

    ###  merging intersections within each type (reducing duplicates/overlaps) ) ###
    chem_spans = merge_overlaps(chem_spans)
    spec_spans = merge_overlaps(spec_spans)

    ### combining the final spans of the two types  ###
    spans = chem_spans + spec_spans

    ### preparing sentences for context (from the species model, where there is a sentencizer) ###
    sentences = list(doc_spec.sents)

    results = []

    ### post-processing each span: text cleaning, length filtering, finding context sentences ###
    for s, e, txt, typ in spans:
        txt_clean = sanitize_text(txt)
        if len(txt_clean) < MIN_LEN:
            continue

        # looking for a sentence that covers the start of the entity
        sent_text = next(
            (sent.text for sent in sentences if sent.start_char <= s <= sent.end_char),
            ""
        )

        ### forming the result record for this span ###
        results.append({
            "entity": txt_clean,
            "type": typ,
            "start": s,
            "end": e,
            "sentence": sent_text
        })

    ### final list of entities ###
    return results
    
    
### TASK (high-level):
# Read an Excel sheet, run two spaCy NER models on each row's text,
# collect CHEMICAL and SPECIES entities with character spans and sentence context,
# print coverage statistics, and save results to CSV/Parquet.

# MAIN CODE 

### 1) Load input Excel ###
# - Validate path
# - Read target sheet
# - Ensure a row identifier column exists
if not os.path.exists(INPUT_XLSX):
    raise FileNotFoundError(f"Excel file not found: {INPUT_XLSX}")

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

if ID_COL not in df.columns:
    df[ID_COL] = range(1, len(df) + 1)

print(f"[i] Loaded {len(df)} rows from sheet '{SHEET_NAME}'")
print(f"[i] Available columns: {list(df.columns)}")

### 1a) Pick the text column to analyze ###
print(f"[i] Using text column: {TEXT_COL}")
print(df[[ID_COL, TEXT_COL]].head(3))

### 2) Load spaCy models ###
# - CHEMICAL model + SPECIES model (adds sentencizer if available)
nlp_chem, nlp_spec = load_models()

### 3) Extract entities for all rows ###
# - Compose text per row
# - Run extract_entities()
# - Normalize and append to results list
results = []
for _, row in df.iterrows():
    text = compose_text(row)
    ents = extract_entities(text, nlp_chem, nlp_spec)
    for e in ents:
        results.append({
            "Paper": row[ID_COL],
            "entity": e["entity"],
            "type": e["type"],              # CHEMICAL or SPECIES
            "char_start": e["start"],
            "char_end": e["end"],
            "context_sentence": e["sentence"]
        })

### 4) Build output DataFrame ###
out_df = pd.DataFrame(results).drop_duplicates()

### 5) Quick summary / sanity checks ###
# - Count by type
# - Coverage (unique papers with ≥1 entity)
# - Top-10 entities per type
if out_df.empty:
    print("\n[!] No entities were found.")
else:
    type_counts = out_df["type"].value_counts().sort_index()
    print("\n=== Entity counts by type ===")
    for t, c in type_counts.items():
        print(f"{t:9s}: {c}")

    covered = out_df.groupby("type")["Paper"].nunique().sort_index()
    print("\n=== Number of unique papers mentioning each type ===")
    for t, n in covered.items():
        print(f"{t:9s}: {n} papers")

    total_rows = len(df)
    rows_with_any = out_df["Paper"].nunique()
    print(f"\nCoverage: {rows_with_any}/{total_rows} papers ({rows_with_any/total_rows:.1%}) contain ≥1 entity")

    print("\n=== Top 10 CHEMICAL entities ===")
    top_chem = (out_df[out_df["type"]=="CHEMICAL"]["entity"]
                .str.strip().str.lower().value_counts().head(10))
    for ent, cnt in top_chem.items():
        print(f"{cnt:5d}  {ent}")

    print("\n=== Top 10 SPECIES entities ===")
    top_species = (out_df[out_df["type"]=="SPECIES"]["entity"]
                   .str.strip().str.lower().value_counts().head(10))
    for ent, cnt in top_species.items():
        print(f"{cnt:5d}  {ent}")

### 6) Persist outputs (CSV/Parquet) ###
out_csv = f"{OUT_PREFIX}.csv"
out_parquet = f"{OUT_PREFIX}.parquet"
out_df.to_csv(out_csv, index=False)
try:
    out_df.to_parquet(out_parquet, index=False)
except Exception:
    pass

print(f"\n[OK] Extraction finished: {len(out_df)} entities")
print(f"CSV saved to: {os.path.abspath(out_csv)}")
if os.path.exists(out_parquet):
    print(f"Parquet saved to: {os.path.abspath(out_parquet)}")


# 7) First rows for quick check 
out_df.head(10)
