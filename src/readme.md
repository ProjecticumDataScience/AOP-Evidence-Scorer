## NER extraction

###  Gene pipline 

this module reads a biomedical sentence and turns it into structured “agent → effect → gene” triples. 
The agent is typically a chemical/drug/diet, the effect is a direction (increase/decrease), and the gene is a human gene normalized to its official symbol. 
The goal is to mine literature and produce edges you can drop into a knowledge graph or downstream analysis.

Pipeline:

Text normalization -> Loads a  gene alias dictionary -> Finds the agent=the cause  ( en_ner_bc5cdr_md). ->Finds genes in the sentence->Infers directions->Builds triples

- Example:  "High uric acid  --increase-->  MAPK8"

## model_train_pytorch
###  This script teaches the model to identify causal relationships based on a given dataset. 
This script teaches the model to identify causal relationships based on a given dataset.
We trained our model on data extracted from the study Optimization of an Adverse Outcome Pathway Network on Chemical-Induced Cholestasis Using an Artificial Intelligence–Assisted 
Data Collection and Confidence Level Quantification Approach, using the paper’s appendices/supplementary tables as the primary  source *
*  https://www.sciencedirect.com/science/article/pii/S1532046423001867  ** PUBLIC ACCESS


We put labeled data 
- [E1]de_novo_lipogenesis_fa_synthesis[/E1] 
- [E2]liver_triglyceride_accumulation[/E2]
  and correlation 
 - "label": 1
  into the model.
  
- Example input data:

```json

  {
    "PMID": 28189721,
    "KE_upstream": "de_novo_lipogenesis_fa_synthesis",
    "KE_downstream": "liver_triglyceride_accumulation",
    "Stressor": "pollutant",
    "Chemical": "1,2_dichloroethane",
    "Species": "mouse",
    "Test_system": "in vivo",
    "Correlation": "negative correlation",
    "text": "These results suggest that hepatic glucose and lipid homeostasis are impaired by 1,2-DCE exposure via down-regulation of PYGL and G6PC expression, which may be primarily mediated by the 2-chloroacetic acid-activated Akt1 pathway. [E1]de_novo_lipogenesis_fa_synthesis[/E1] decreases [E2]liver_triglyceride_accumulation[/E2]. [E1]de_novo_lipogenesis_fa_synthesis[/E1] decreases [E2]liver_triglyceride_accumulation[/E2].",
    "Evidence_level": "translational;transcriptional",
    "label": 1
  }
  
  ```
  
  we expect the model to lean the correlation between E1 and E2 in the similar context


## logistic_regession_stressor.py :  logistic regression to classify  stressor  from given chemical/drug/diet

This script trains a text classifier to predict a chemical’s stressor category, then applies it to another CSV and saves the results. 

Loads data_input/ker_dataset_table_s2.json, keeps Chemical and Stressor, and uses Chemical as input X and Stressor as target y. 

Builds a scikit-learn Pipeline:

TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), ...) converts each chemical name into TF-IDF features based on character n-grams (3–6 chars).

LogisticRegression(max_iter=2000, class_weight="balanced") learns to map those features to the stressor labels. 

Fits the model on the full JSON dataset. 

Reads data_input/chem_relations.csv, predicts a new column stressor_pred from its chemical column (filling missing values with empty strings). 

Writes the enriched file to data_output/chem_relations_with_stressor.csv and prints the first 10 rows of PMID, chemical, and stressor_pred.



