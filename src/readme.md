## model_train_pytorch
###  This script teaches the model to identify causal relationships based on a given dataset. 
This script teaches the model to identify causal relationships based on a given dataset. For example:
nuclear_receptor_changes_pparg->de_novo_lipogenesis_fa_synthesis

We put labeled data 
- [E1]de_novo_lipogenesis_fa_synthesis[/E1] 
- [E2]liver_triglyceride_accumulation[/E2]
  and correlation 
 - "label": 1
  into the model.


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
