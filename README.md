
# Adverse Outcome Pathways

This repository focuses on entity extraction and biological plausibility evaluation from scientific articles related to Adverse Outcome Pathways (AOP).  
It processes curated data about Key Event Relationships (KERs), extracts relevant biological entities, and evaluates mechanistic plausibility.

### Weight of Evidence (WoE)

The project also performs **Weight of Evidence (WoE) analysis** for each KER,
 summarizing and scoring the strength of the evidence (high / moderate / low) based on reproducibility, 
consistency, and relevance of the supporting data.
In practical terms:
For each connection (Key Event Relationship, KER), publications, experiments, and observations are gathered.
The quality of this data is assessed: how representative, consistent, and reproducible it is.
A confidence level is assigned (high / moderate / low).
---

##  Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure) 
- [Folders](#folders)
- [Workflow](#workflow)

---

## Overview

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
