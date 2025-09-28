output data
================

### entities_ner.csv

This file contains the **results of the Named Entity Recognition (NER)
pipeline** applied to the pre-filtered set of articles.  
It includes all **extracted entities** relevant to Adverse Outcome
Pathways (AOP), such as:

- **Species** – e.g., mouse, rat, human  
- **Stressors** – chemicals, pollutants, pharmaceuticals  
- **Key Events / Biological Processes** – mechanistic events mentioned
  in the literature

### example output data

``` csv
Paper,entity,type,char_start,char_end,context_sentence

/work/eval_papers/PMID_29173234.pdf,Mouse,SPECIES,496,501,"- **Species:** Mouse - **Species Snippet:** Male Balb/c mice fed with diets containing α-lactalbumin developed abdominal adiposity, insulin resistance, and moderate liver steatosis, with increased hepatic lipid content."
```

## Explanation of fields

| Column | Meaning |
|:---|:---|
| Paper | `/work/eval_papers/PMID_29173234.pdf` → the source document (PDF). |
| entity | `Mouse` → the detected mention in text. |
| type | `SPECIES` → NER label from the `en_ner_bionlp13cg_md` model. |
| char_start | `496` → character offset where the entity begins. |
| char_end | `501` → character offset where it ends. |
| context_sentence | The full sentence containing the mention of the mouse, including context describing what happened to it (adiposity, insulin resistance, steatosis). |
