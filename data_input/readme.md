
input data
================

### Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx

This Excel file contains 97 pre-filtered scientific articles selected for their relevance to Adverse Outcome Pathways (AOP).  
These articles form the foundation for all subsequent analysis, including:

- Entity extraction – identifying species, stressors, and key events from the texts  
- Biological plausibility evaluation – assessing mechanistic links between key events  

This dataset serves both as input for the pipeline and as the benchmark for validating results.


### example output data

``` csv
de_novo_lipogenesis_fa_synthesis-novel_oxidative_stress /work/eval_papers/PMID_29704577.pdf 6   1   2025-06-09  openai/o4-mini-high mouse   pharmaceutical  in vivo functional;transcriptional  positive    Seventy-two C57BL/6 mice were equally randomized into six groups and treated with a standard diet (SD) or high-fat diet (HFD) alone or combined with testosterone cypionate (10 or 20 mg/kg) for 12 weeks. When combined with a HFD, AS reduced plasma HDL cholesterol levels. It also upregulated SREBP-1, PPARα, SCD-1 and ACOX1 gene expression; plasma and hepatic triglyceride levels; oxidative stress; circulating hepatic transaminase levels and NAFLD severity.   Remains unknown if dietary lipids and anabolic steroids (AS) can interact to modify energy metabolism, hepatic structure and function. We investigated the impact of AS on gene expression, lipid profile, redox status and the development of nonalcoholic fatty liver disease (NAFLD) in mice treated with a diet rich in trans fatty acids. Seventy-two C57BL/6 mice were equally randomized into six groups and treated with a standard diet (SD) or high-fat diet (HFD) alone or combined with testosterone cypionate (10 or 20 mg/kg) for 12 weeks.   Seventy-two C57BL/6 mice were equally randomized into six groups and treated with a standard diet (SD) or high-fat diet (HFD) alone or combined with testosterone cypionate (10 or 20 mg/kg) for 12 weeks. When combined with a HFD, AS reduced plasma HDL cholesterol levels. It also upregulated SREBP-1, PPARα, SCD-1 and ACOX1 gene expression; plasma and hepatic triglyceride levels; oxidative stress; circulating hepatic transaminase levels and NAFLD severity.   We investigated the impact of AS on gene expression, lipid profile, redox status and the development of nonalcoholic fatty liver disease (NAFLD) in mice treated with a diet rich in trans fatty acids. qPCR analysis revealed that all groups treated with AS or a HFD exhibited increased PPARα, ACOX1, SREBP-1 and SCD-1 gene expression levels when compared to animals receiving a SD (P < 0.05). It also measured plasma and hepatic triglyceride levels, oxidative stress markers, circulating hepatic transaminase levels and NAFLD severity, providing functional evidence.    qPCR analysis revealed that all groups treated with AS or a HFD exhibited increased PPARα, ACOX1, SREBP-1 and SCD-1 gene expression levels when compared to animals receiving a SD (P < 0.05). In general, expression of these genes was higher in the HFD + TC10 and HFD + TC20 groups compared to the other groups (P < 0.05). In animals treated with a SD or HFD, AS-induced gene expression showed a dose-dependent behavior (Fig. 3). "The biological plausibility evaluation for the KER ""de_novo_lipogenesis_fa_synthesis-novel_oxidative_stress"" based on the provided paper yields the following results:

- **Biological Plausibility Score:** 6.0

**Supporting Evidence from the Paper (/work/eval_papers/PMID_29704577.pdf):**

**Stressor:** Pharmaceutical (Testosterone cypionate combined with high-fat diet)  
**Stressor Snippet:** Remains unknown if dietary lipids and anabolic steroids (AS) can interact to modify energy metabolism, hepatic structure and function. The study investigated the impact of AS on gene expression, lipid profile, redox status, and NAFLD development in mice treated with a diet rich in trans fatty acids.  
**Species:** Mouse  
**Species Snippet:** 72 C57BL/6 mice were randomized into groups and treated with diets (standard or high-fat) with or without testosterone cypionate for 12 weeks.  
**Test System:** In vivo mouse model  
**Evidence Level:** Functional; Transcriptional  
**Evidence Snippet:** All groups treated with AS or high-fat diet exhibited increased PPARα, ACOX1, SREBP-1, and SCD-1 gene expression, along with elevated triglyceride levels, oxidative stress markers, and increased NAFLD severity, indicating metabolic alterations related to lipid synthesis and oxidative stress.  
**Correlation:** Positive  
**Correlation Snippet:** Increased expression of lipid synthesis genes in response to AS and high-fat diet, showing a dose-dependent relationship, supports the link between lipid synthesis pathways and oxidative stress.

This evidence supports a positive correlation between de novo lipogenesis and oxidative stress, aligning with the KER, and supports its biological plausibility in this context."

```

## Explanation of fields

| Field | Value |
|:---|:---|
| KER | `de_novo_lipogenesis_fa_synthesis-novel_oxidative_stress` |
| Paper | `/work/eval_papers/PMID_29704577.pdf` |
| Biological Plausibility Score | `6.0` |
| Run / Date / Model | `#1` · `2025-06-09` · `openai/o4-mini-high` |
| Species | `mouse` |
| Stressor type | `pharmaceutical` (testosterone cypionate + high-fat diet) |
| Test system | `in vivo` |
| Evidence level | `functional; transcriptional` |
| Correlation | `positive` |
| Stressor snippet | Remains unknown if dietary lipids and anabolic steroids (AS) can interact… The study investigated AS + HFD for 12 weeks. |
| Species snippet | 72 C57BL/6 mice randomized; SD or HFD ± testosterone cypionate for 12 weeks. |
| Evidence snippet | Increased PPARα, ACOX1, SREBP-1, SCD-1; higher TG; oxidative stress; transaminases; NAFLD severity. |
| Correlation snippet | Dose-dependent increases in lipid-synthesis genes with AS + HFD support the link to oxidative stress. |
| Conclusion | Evidence supports a **positive** correlation between de novo lipogenesis and oxidative stress, aligning with the KER. |
