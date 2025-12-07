# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) Train the model in the same way as in chemical-stressor.html
train = pd.read_json("data_input/ker_dataset_table_s2.json")
train = train[["Chemical", "Stressor"]].dropna()

X = train["Chemical"].astype(str)
y = train["Stressor"].astype(str)

clf = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=1, max_df=0.9)),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

clf.fit(X, y)

#2) We read your results file and predict the stressor based on the chemical column
res = pd.read_csv("data_input/chem_relations.csv")

res["stressor_pred"] = clf.predict(res["chemical"].fillna("").astype(str))

#3) Save a new CSV with the predicted stressor
res.to_csv("data_output/chem_relations_with_stressor.csv", index=False)

print(res[["PMID", "chemical", "stressor_pred"]].head(10))

