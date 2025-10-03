## model_train_pytorch
###  This script teaches the model to identify causal relationships based on a given dataset. 
This script teaches the model to identify causal relationships based on a given dataset. For example:
nuclear_receptor_changes_pparg->de_novo_lipogenesis_fa_synthesis

We put labeled data 
-[E1]de_novo_lipogenesis_fa_synthesis[/E1] 
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
    "text": "1,2-Dichloroethane impairs glucose and lipid homeostasis in the livers of NIH Swiss mice. Zeng N(1), . Author information: (1)Faculty of Preventive Medicine, A Key Laboratory of Guangzhou Environmental Pollution and Risk Assessment, School of Public Health, Sun Yat-sen University, Guangzhou 510080, China. (2)Guangdong Provincial Key Laboratory of Occupational Disease Prevention and Treatment, Department of Toxicology, Guangdong Province Hospital for Occupational Disease Prevention and Treatment, Guangzhou 510300, China. (3)Guangdong Provincial Key Laboratory of Occupational Disease Prevention and Treatment, Department of Toxicology, Guangdong Province Hospital for Occupational Disease Prevention and Treatment, Guangzhou 510300, China. Electronic address: huangzhenlie@126.com. (4)Faculty of Preventive Medicine, A Key Laboratory of Guangzhou Environmental Pollution and Risk Assessment, School of Public Health, Sun Yat-sen University, Guangzhou 510080, China. Electronic address: wangq27@mail.sysu.edu.cn. Excessive exposure to 1,2-Dichloroethane (1,2-DCE), a chlorinated organic toxicant, can lead to liver dysfunction. To fully explore the mechanism of 1,2-DCE-induced hepatic abnormalities, 30 male National Institutes of Health (NIH) Swiss mice were exposed to 0, 350, or 700mg/m3 of 1,2-DCE, via inhalation, 6h/day for 28days. Increased liver/body weight ratios, as well as serum AST and serum ALT activity were observed in the 350 and 700mg/m3 1,2-DCE exposure group mice, compared with the control group mice. In addition, decreased body weights were observed in mice exposed to 700mg/m3 1,2-DCE, compared with control mice. Exposure to 350 and 700mg/m3 1,2-DCE also led to significant accumulation of hepatic glycogen, free fatty acids (FFA) and triglycerides, elevation of blood triglyceride and FFA levels, and decreases in blood glucose levels. Results from microarray analysis indicated that the decreases in glucose-6-phosphatase catalytic subunit (G6PC) and liver glycogen phosphorylase (PYGL) expression, mediated by the activation of AKT serine/threonine kinase 1 (Akt1), might be responsible for the hepatic glycogen accumulation and steatosis. Further in vitro study demonstrated that 2-chloroacetic acid (1,2-DCE metabolite), rather than 1,2-DCE, up-regulated Akt1 phosphorylation and suppressed G6PC and PYGL expression, resulting in hepatocellular glycogen accumulation. These results suggest that hepatic glucose and lipid homeostasis are impaired by 1,2-DCE exposure via down-regulation of PYGL and G6PC expression, which may be primarily mediated by the 2-chloroacetic acid-activated Akt1 pathway. [E1]de_novo_lipogenesis_fa_synthesis[/E1] decreases [E2]liver_triglyceride_accumulation[/E2]. [E1]de_novo_lipogenesis_fa_synthesis[/E1] decreases [E2]liver_triglyceride_accumulation[/E2].",
    "Evidence_level": "translational;transcriptional",
    "label": 1
  }
  
  ```
  
  we acpect the model to lean the correlation between E1 and E2 in the similar context