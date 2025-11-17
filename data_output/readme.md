output data
================

### relations.txt

From 92 articles on steatosis, 629 relations were extracted.

This file contains the **results of the relation extraction pipeline**
applied to the pre-filtered set of articles.
It includes all **chemical–event relations** relevant to Adverse Outcome
Pathways (AOP), such as:

- **Chemicals / Stressors** – e.g., fructose, ethanol, PFOS  
- **Triggers (relation phrases)** – e.g., induce, prevent, attenuate, decrease  
- **Targets (outcomes / key events)** – e.g., insulin resistance, liver fat accumulation, hepatic steatosis  

### example output data

``` tsv
PMID	chemical	trigger	target
25736031	Fructose	induce	insulin resistance
```

## Explanation of fields

| Column  | Meaning |
|:-------|:--------|
| PMID    | PubMed identifier of the article where the relation was extracted. |
| chemical | Name of the stressor / compound mentioned in the sentence (e.g., Fructose, ethanol, PFOS). |
| trigger  | The verb or short phrase describing the effect of the chemical (e.g., induce, prevent, decrease). |
| target   | The biological effect, phenotype, or process affected by the chemical (e.g., insulin resistance, liver fat accumulation, hepatic steatosis). |

---

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

PMID_29173234.pdf,SPECIES,496,501,"- **Species:** Homo Sapiens - **Species Snippet:** SIRT1 disruption in human fetal hepatocytes elevates intracellular glucose and lipids via increased lipogenesis/gluconeogenesis and reduced AKT/FOXO1 signaling."
```

## Explanation of fields

| Column | Meaning |
|:---|:---|
| Paper | `PMID_29173234.pdf` → the source document (PDF). |
| entity | ` SIRT1 disruption ` → the detected mention in text. |
| type | `Chemicals` → NER label from the `en_ner_bionlp13cg_md` model. |
| char_start | `496` → character offset where the entity begins. |
| char_end | `501` → character offset where it ends. |
| context_sentence | The full sentence containing the mention of the homo sapiens  including context describing what happened to it (adiposity, insulin resistance, steatosis). |

