import time, json, pickle, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ====================
# Загружаем данные
# ====================
df = pd.read_json("data_input/dataset.json")  # твой JSON с text

# Кодируем нужные поля
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

# сохраняем энкодеры
os.makedirs("models", exist_ok=True)
pickle.dump(enc_up, open("models/enc_up.pkl","wb"))
pickle.dump(enc_down, open("models/enc_down.pkl","wb"))
pickle.dump(enc_corr, open("models/enc_corr.pkl","wb"))
pickle.dump(enc_species, open("models/enc_species.pkl","wb"))
pickle.dump(enc_system, open("models/enc_system.pkl","wb"))

# ====================
# Dataset
# ====================
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class KEDataset(Dataset):
    def __init__(self, df, max_len=256):
        self.df = df
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = tokenizer(r["text"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "y_up": torch.tensor(r["y_up"]),
            "y_down": torch.tensor(r["y_down"]),
            "y_corr": torch.tensor(r["y_corr"]),
            "y_species": torch.tensor(r["y_species"]),
            "y_system": torch.tensor(r["y_system"])
        }

# ====================
# Model
# ====================
class KEClassifier(nn.Module):
    def __init__(self, name, nu, nd, nc, ns, nts, freeze=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        if freeze:
            for p in self.bert.parameters(): p.requires_grad=False
        h = self.bert.config.hidden_size
        self.up = nn.Linear(h,nu)
        self.down = nn.Linear(h,nd)
        self.corr = nn.Linear(h,nc)
        self.species = nn.Linear(h,ns)
        self.system = nn.Linear(h,nts)
    def forward(self, ids, mask):
        o = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:,0,:]
        return {
            "up": self.up(o),
            "down": self.down(o),
            "corr": self.corr(o),
            "species": self.species(o),
            "system": self.system(o)
        }

# ====================
# Обучение
# ====================
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

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0
    for i,b in enumerate(loader):
        opt.zero_grad()
        out = model(b["input_ids"], b["attention_mask"])
        loss = (
            F.cross_entropy(out["up"], b["y_up"]) +
            F.cross_entropy(out["down"], b["y_down"]) +
            F.cross_entropy(out["corr"], b["y_corr"]) +
            F.cross_entropy(out["species"], b["y_species"]) +
            F.cross_entropy(out["system"], b["y_system"])
        )
        loss.backward(); opt.step()
        total_loss += loss.item()

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            iters_left = len(loader) - (i+1)
            eta = elapsed / (i+1) * iters_left
            print(f"Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(loader)}, Loss={loss.item():.4f}, ETA {eta/60:.1f} min")

    print(f"Epoch {epoch+1} done. Avg loss={total_loss/len(loader):.4f}")

# ====================
# Сохраняем модель
# ====================
torch.save(model.state_dict(),"models/pubmedbert_cpu_clean.pt")
print("✅ Модель и энкодеры сохранены в папку models/")
