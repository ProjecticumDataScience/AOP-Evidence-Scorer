### `entities_ner.csv`

This file contains the **results of the Named Entity Recognition (NER) pipeline** applied to the pre-filtered set of articles.  
It includes all **extracted entities** relevant to Adverse Outcome Pathways (AOP), such as:

- **Species** – e.g., mouse, rat, human  
- **Stressors** – chemicals, pollutants, pharmaceuticals  
- **Key Events / Biological Processes** – mechanistic events mentioned in the literature  


### example output data

/work/eval_papers/PMID_29173234.pdf,
Mouse,
SPECIES,
496,
501,
"- **Species:** Mouse - **Species Snippet:** Male Balb/c mice fed with diets containing α-lactalbumin developed abdominal adiposity, insulin resistance, and moderate liver steatosis, with increased hepatic lipid content."

```csv
/work/eval_papers/PMID_29173234.pdf,
```
```csv
Mouse,
```
```csv
SPECIES,
```
```csv
496,
```
```csv
501,
```

```csv
"- **Species:** Mouse - **Species Snippet:** Male Balb/c mice fed with diets containing α-lactalbumin developed abdominal adiposity, insulin resistance, and moderate liver steatosis, with increased hepatic lipid content."
```

### Explanation of fields

```{r, echo=FALSE, message=FALSE, warning=FALSE}
library(tibble)
library(knitr)
library(kableExtra)

tbl <- tribble(
  ~Column,            ~Meaning,
  "Paper",            "`/work/eval_papers/PMID_29173234.pdf` \u2192 the source document (PDF).",
  "entity",           "`Mouse` \u2192 the detected mention in text.",
  "type",             "`SPECIES` \u2192 NER label from the `en_ner_bionlp13cg_md` model.",
  "char_start",       "`496` \u2192 character offset where the entity begins.",
  "char_end",         "`501` \u2192 character offset where it ends.",
  "context_sentence", "The full sentence containing the mention of the mouse, including context describing what happened to it (adiposity, insulin resistance, steatosis)."
)

kable(tbl, format = "html", escape = FALSE) |>
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover"))
```

