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


### chem_relations.tsv / chem_relations.csv

This file contains the **results of the chemical relation extraction
pipeline** applied to the pre-filtered set of PubMed articles on
steatosis and related liver outcomes. Each row corresponds to one
chemical–event relation found in an abstract.

The data can be exported either as a tab‑separated file
`chem_relations.tsv` or as a comma‑separated file `chem_relations.csv`.
The two files contain exactly the same columns; only the delimiter
differs.

### Example rows

```tsv
PMID	chemical	trigger	target	context_sentence
25736031	Fructose	induce	insulin resistance	Fructose diets have been shown to induce insulin resistance and to alter liver metabolism and gut barrier function, ultimately leading to non-alcoholic fatty liver disease.
25736031	Citrulline	improve	insulin sensitivity	Citrulline, Glutamine and Arginine may improve insulin sensitivity and have beneficial effects on gut trophicity.
```

### Explanation of fields

| Column           | Meaning |
|:-----------------|:--------|
| PMID             | PubMed identifier of the article where the relation was extracted. |
| chemical         | Name of the chemical / stressor mentioned in the sentence (e.g., Fructose, ethanol, PFOS). |
| trigger          | Normalized predicate describing the effect of the chemical (e.g., `induce`, `decrease`, `prevent`, or more complex phrases such as `plays an important role in`). |
| target           | Biological effect, phenotype or key event that is affected by the chemical (e.g., `insulin resistance`, `liver fat accumulation`, `endotoxemia`). In cases where the verb is too generic (for example *plays an important role*), simple rule‑based post‑processing extends the phrase using the local sentence context to make the trigger–target pair more informative. |
| context_sentence | Full sentence from the abstract where the relation was detected. This snippet provides the original textual evidence for the relation. Any tab or newline characters are removed so that the field is safe to use in TSV/CSV format. |

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

PMID_26890260.pdf,Homo sapiens,SPECIES,496,501,"- **Species:** Homo sapiens - **Species Snippet:** In human fetal hepatocytes, inhibition of SIRT1 increases intracellular glucose and lipid accumulation and activates de novo lipogenesis and gluconeogenesis."

```

## Explanation of fields

| Column | Meaning |
|:---|:---|

| Paper  | `PMID_26890260.pdf` → the source document (PDF) for [PubMed:26890260](https://pubmed.ncbi.nlm.nih.gov/26890260/). |
| species | `Homo sapiens` → the detected mention in text. |
| entity | ` SIRT1 disruption ` → the detected mention in text. |
| type | `Chemicals` → NER label from the `en_ner_bionlp13cg_md` model. |
| char_start | `496` → character offset where the entity begins. |
| char_end | `501` → character offset where it ends. |
| context_sentence | The full sentence containing the mention of the homo sapiens  including context describing what happened to it (adiposity, insulin resistance, steatosis). |

