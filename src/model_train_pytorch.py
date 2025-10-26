"""
Train a multitask text classifier (built on PubMedBERT) that reads a sentence and predicts five labels at once:
upstream KE - downstream KE - correlation - species - test system. 

The model uses the [CLS] embedding from PubMedBERT and five parallel linear heads.

What each function/block is for

- Data loading & encoding
  - pd.read_json("data_input/dataset.json"): loads input  texts and target columns (We trained our model on data extracted from the study Optimization of an Adverse Outcome Pathway Network on Chemical-Induced Cholestasis Using an Artificial Intelligence–Assisted Data Collection and Confidence Level Quantification Approach, using the paper’s appendices/supplementary tables as the primary source*)
  * https://www.sciencedirect.com/science/article/pii/S1532046423001867
  - Five LabelEncoders (enc_up, enc_down, enc_corr, enc_species, enc_system) turn each categorical target into integer IDs; transformed columns are stored as y_up, y_down, y_corr, y_species, y_system.

- Tokenizer setup
  - AutoTokenizer.from_pretrained(MODEL_NAME): builds a tokenizer for BiomedNLP-PubMedBERT-base-uncased-abstract, used to convert text into input IDs and attention masks.

- KEDataset (PyTorch Dataset)
  - __len__: returns dataset size.
  - __getitem__: tokenizes one sample (max_length=256) and returns tensors: input_ids, attention_mask, plus the five integer labels. This is what the DataLoader will batch.

- KEClassifier (the model)
  - __init__(name, nu, nd, nc, ns, nts, freeze=True): loads the PubMedBERT encoder; optionally freezes it (default True), then defines five linear “heads” with output sizes matching the number of classes for each task.
  - forward(ids, mask): runs PubMedBERT, takes the [CLS] vector, and computes logits for each of the five tasks (up, down, corr, species, system). 

- Training prep
  - Creates a DataLoader (batch size 4), instantiates KEClassifier with class counts, and sets up Adam optimizer (lr=1e-5). With freeze=True, only the heads are trained. 

- Training loop
  - For each batch: forward pass → sum of five cross-entropy losses → backprop → optimizer step. Prints progress and an ETA every 50 steps; reports average epoch loss.

- Saving artifacts
  - torch.save(model.state_dict(), "2models/pubmedbert_cpu_clean.pt"): saves the trained weights; encoders were saved earlier as enc_*.pkl.


"""



import time, json, pickle, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# DATA LOADING & PREPROCESSING
# Goal: read the dataset and encode categorical targets into integers.

df = pd.read_json("data_input/dataset.json")  # your JSON with "text" and labels

# Encode required fields
enc_up = LabelEncoder().fit(df["KE_upstream"])
enc_down = LabelEncoder().fit(df["KE_downstream"])
enc_corr = LabelEncoder().fit(df["Correlation"])
enc_species = LabelEncoder().fit(df["Species"])
enc_system  = LabelEncoder().fit(df["Test_system"])

df["y_up"] = enc_up.transform(df["KE_upstream"])
df["y_down"] = enc_down.transform(df["KE_downstream"])
df["y_corr"] = enc_corr.transform(df["Correlation"])
df["y_species"] = enc_species.transform(df["Species"])
df["y_system"]  = enc_system.transform(df["Test_system"])

# Persist encoders to reuse at inference time
os.makedirs("models", exist_ok=True)
pickle.dump(enc_up, open("2models/enc_up.pkl","wb"))
pickle.dump(enc_down, open("2models/enc_down.pkl","wb"))
pickle.dump(enc_corr, open("2models/enc_corr.pkl","wb"))
pickle.dump(enc_species, open("2models/enc_species.pkl","wb"))
pickle.dump(enc_system, open("2models/enc_system.pkl","wb"))


# TOKENIZER SETUP
# Goal: prepare the tokenizer from the chosen transformer model.

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# DATASET DEFINITION
# Goal: wrap the dataframe into a PyTorch Dataset that tokenizes text and returns labels.


class KEDataset(Dataset):
    def __init__(self, df, max_len=256):
        # Inside function: store dataframe and configs.
        self.df = df
        self.max_len = max_len

    def __len__(self):
        # Inside function: return dataset size.
        return len(self.df)

    def __getitem__(self, i):
        # Inside function: tokenize text and return tensors for inputs and labels.
        r = self.df.iloc[i]
        x = tokenizer(
            r["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "y_up": torch.tensor(r["y_up"]),
            "y_down": torch.tensor(r["y_down"]),
            "y_corr": torch.tensor(r["y_corr"]),
            "y_species": torch.tensor(r["y_species"]),
            "y_system": torch.tensor(r["y_system"])
        }


# MODEL DEFINITION
# Goal: build a multitask classifier on top of PubMedBERT [CLS] representation.

class KEClassifier(nn.Module):
    def __init__(self, name, nu, nd, nc, ns, nts, freeze=True):
        # Inside function: initialize encoder and task-specific heads.
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        h = self.bert.config.hidden_size
        self.up = nn.Linear(h, nu)
        self.down = nn.Linear(h, nd)
        self.corr = nn.Linear(h, nc)
        self.species = nn.Linear(h, ns)
        self.system = nn.Linear(h, nts)

    def forward(self, ids, mask):
        # Inside function: get [CLS] embedding and compute logits for each task.
        o = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        return {
            "up": self.up(o),
            "down": self.down(o),
            "corr": self.corr(o),
            "species": self.species(o),
            "system": self.system(o)
        }


# TRAINING PREP
# Goal: create dataloader, instantiate the model, and set the optimizer/hyperparameters.

device = "cpu"
dataset = KEDataset(df)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = KEClassifier(
    MODEL_NAME,
    len(enc_up.classes_), len(enc_down.classes_), len(enc_corr.classes_),
    len(enc_species.classes_), len(enc_system.classes_)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS = 3


# TRAINING LOOP
# Goal: train the model with a summed cross-entropy loss across tasks and report progress.

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0
    for i, b in enumerate(loader):
        # Inside loop: forward, compute multitask loss, backprop, optimizer step.
        opt.zero_grad()
        out = model(b["input_ids"], b["attention_mask"])
        loss = (
            F.cross_entropy(out["up"], b["y_up"]) +
            F.cross_entropy(out["down"], b["y_down"]) +
            F.cross_entropy(out["corr"], b["y_corr"]) +
            F.cross_entropy(out["species"], b["y_species"]) +
            F.cross_entropy(out["system"], b["y_system"])
        )
        loss.backward()
        opt.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            iters_left = len(loader) - (i + 1)
            eta = elapsed / (i + 1) * iters_left
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{len(loader)} | "
                f"Loss={loss.item():.4f} | ETA ~ {eta/60:.1f} min"
            )

    print(f"Epoch {epoch+1} complete. Average loss = {total_loss/len(loader):.4f}")


# SAVE ARTIFACTS
# Goal: persist the trained model weights and previously saved encoders.

torch.save(model.state_dict(), "2models/pubmedbert_cpu_clean.pt")
print("Model and encoders have been saved to the '2models/' directory.")

