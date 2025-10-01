
# Adverse Outcome Pathways

This project uses deep learning (ML) to predict toxic effects in biology. 
The starting point of this project is the research by Anouk Verhoeven (Marc Teunes):
"A quantitative weight-of-evidence method for confidence assessment of 
adverse outcome pathway networks: A case study on chemical-induced 
liver steatosis ".
It focuses on entity extraction and biological plausibility evaluation from scientific articles related 
to Adverse Outcome Pathways (AOP).  
It processes curated data about Key Event Relationships (KERs), 
extracts relevant biological entities, and evaluates mechanistic plausibility.



##  Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure) 
- [Folders](#folders)
- [Workflow](#workflow)

---

## Overview


### Weight of Evidence (WoE)

This project includes Weight of Evidence (WoE) analysis for each KER.  
Evidence strength is scored high / moderate / low based on reproducibility, consistency, and relevance of the supporting studies.

The project enables the extraction of key biological entities (e.g., stressor, species, evidence levels, correlations) and the evaluation of biological plausibility between events.  
It leverages manually curated ground-truth data to validate the performance of an automated pipeline and stores results in a structured format.

---

## Repository Structure

- **data_input/** – Input data for entity extraction & evaluation (`Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.csv`) 
- **data_output/** – Contains processed results (e.g., `entities_ner.csv`, `entities_ner.parquet`)  
- **src/** – Source code for entity extraction pipeline (`extract_entities_ner.py`)
## Folders

### `data_input/`
Holds all raw input data for the pipeline.  
Currently includes the Excel file described above, which provides the articles and curated KERs used for downstream entity extraction and plausibility evaluation.

### `data_output/`
Contains processed results after running the pipeline:
- `entities_ner.csv` – Extracted entities in CSV format for quick inspection.
- `entities_ner.parquet` – Same data in Parquet format for efficient processing.

### `src/`
Source code for the entity extraction pipeline:
- `extract_entities_ner.py` – Implements the Named Entity Recognition (NER) workflow, processing article text to extract species, stressors, and key events, and saving them to `data_output/`.

---

## Workflow

### Step 1. Extracting causal relationships


In the first stage, we use the appendices of the article by  Anouk Verhoeven,  
which describe cause–effect relationships.



 
 - **Cause (`up_stream`)**: `liver_triglyceride_accumulation` — accumulation of triglycerides in the liver  
- **Effect (`down_stream`)**: `steatosis` — development of fatty liver disease (hepatic steatosis)  
- **Correlation (`correlation`)**: `positive correlation` — a positive relationship, meaning that the more triglycerides accumulate, the higher the likelihood of steatosis 

 From this data, we prepare a training set for the model using tensors (deep learning). 
 After that, we train the model to find causal relationships in the context of steatosis in new publications/research.
 
 link for the article:
 
 [link for the article:](https://www.sciencedirect.com/science/article/abs/pii/S0300483X24000957)
 
 
 ###  After training the model, we get the results for new articles like this:
 
 ```json
{
  "PMID": 29704577,
  "KE_upstream": "nuclear_receptor_changes_pparg",
  "KE_downstream": "liver_triglyceride_accumulation",
  "Stressor": "food supplement",
  "Chemical": "fructose",
  "Species": "mouse",
  "Test_system": "in vitro",
  "Correlation": "positive correlation",
  "text": "Rats were fed a high-fructose diet (20% fructose) which induced \nhepatic triglyceride accumulation. Exposure to 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD) also caused liver injury."
}

```
### Step 2. Extracting metadata(stressor, chemical) with classic NER  approach


After training the model, it can identify cause-and-effect relationships in the context of liver steatosis.
To extract "Chemical" from the context, we use a classic NER  approach (scispacy, the biomedical model BC4CHEMD).
Next, we classify this chemical using existing databases ("Stressor": "food supplement").
For example: 


 ```json
{	
	"Chemical": "fructose",
	"Stressor": "food supplement"
}
```	

### Step 3. Applying Weight of Evidence 


Applying Weight of Evidence (WoE) based on the created dataset to quantitatively demonstrate how plausible the causal relationship KE1 → KE2 is from a biological perspective, and to compare the strength of different KERs within AOP.
Example

- KER "de novo lipogenesis → triglyceride accumulation":

is supported by dozens of articles, in various forms (rat, mouse, human), with different stressors (alcohol, BPA, high-fat diet), and at different levels (transcriptional, functional).

The BPscore will be high.

- KER "mitochondrial swelling → liver fibrosis":

is supported by 1 article, only in vitro, and only in mice.

The BPscore will be low.