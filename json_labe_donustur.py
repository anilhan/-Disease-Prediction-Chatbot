# 2. Label-to-Class JSON Kaydetme
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

df = pd.read_csv("data/onislenmis_bert_turk_veriseti.csv", sep=";", encoding="utf-8-sig")
label_encoder = LabelEncoder()
df["Hastalik"] = df["label"].map(df.drop_duplicates("label")["Hastalık"])
label_encoder.fit(df["Hastalık"])
id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

with open("data/id_to_label.json", "w", encoding="utf-8-sig") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=2)
