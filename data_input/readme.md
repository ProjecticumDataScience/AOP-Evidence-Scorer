input data
================

### dataset.json

This file contains  JSON array of records with experimental evidence on AOP relationships for fat metabolism in the liver. Each record corresponds to one 
- KE_upstream → KE_downstream 

- pair with an annotation from a specific PubMed article  Supplementary material. *** PUBLIC ACCESS



``` csv
 https://www.sciencedirect.com/science/article/pii/S1532046423001867
```

## Example input data

| Column | Meaning |
|:---|:---|
| PMID | 34344994 |
| KE_upstream | peroxisomal_beta_oxidation |
| KE_downstream | liver_triglyceride_accumulation |
| Stressor | pollutant |
| Chemical | 2,3,7,8_tetrachlorodibenzo_p_dioxin\_(tcdd) |
| Species | mouse |
| Test_system | in vivo |
| Correlation | negative correlation |
| text | 2021 Aug 3;11(1):15689. Thioesterase induction by 2,3,7,8-tetrachlorodibenzo-p-dioxin results in a futile cycle that inhibits hepatic β-oxidation. … (полный текст можно оставить целиком) |
| Evidence_level | functional;transcriptional;translational;transcriptional;translational;functional;translational |

## Explanation input data

| Column | Meaning |
|:---|:---|
| PMID | The article’s unique PubMed identifier (used to link to the source). |
| KE_upstream | The initiating key event (biological change/process) at the start of the AOP linkage. |
| KE_downstream | The subsequent/target key event (often a phenotypic outcome) in the AOP linkage |
| Stressor | The class of stressor (e.g., pollutant, food supplement, pharmaceutical) |
| Chemical | The specific compound or agent acting as the stressor (use a standard name/ID where possible). |
| Species | The organism in which the evidence was obtained (mouse/rat/human, etc.). |
| Test_system | Experimental system type: in vivo / in vitro / ex vivo / in silico. |
| Correlation | Direction of association between KE_upstream and KE_downstream (e.g., positive correlation, negative correlation). |
| text | A snippet/summary from the paper describing the observation/mechanism (can be a full paragraph) |
| Evidence_level | Evidence tiers, separated by : functional — Phenotypic/biochemical endpoints.; transcriptional — Gene-expression changes (RNA-seq, qPCR);translational — Protein-level changes (proteomics, Western blot).|

``` csv

```

---
---



### Gene_dict.tsv



Tabular dictionary of genes  (from NCBI database) for normalizing mentions in the text (used by NER/mapping). Format: TSV (tab-separated), UTF-8, one line per gene.


``` csv
9606	10	NAT2	AAC2|NAT-2|NAT2|PNAT	N-acetyltransferase 2
```


