import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# İlk sütun hastalık, geri kalanlar belirtiler
disease_col = df.columns[0]
symptom_cols = df.columns[1:]

# Yeni liste oluştur: her satırda hastalık ve ona karşılık gelen belirtiler (1 olanlar)
converted_data = []

for idx, row in df.iterrows():
    disease = row[disease_col]
    symptoms = [symptom for symptom in symptom_cols if row[symptom] == 1]
    symptom_str = ",".join(symptoms)
    converted_data.append([disease, symptom_str])

# Yeni dataframe oluştur
new_df = pd.DataFrame(converted_data, columns=["disease", "symptom"])

# Yeni CSV olarak kaydet
new_df.to_csv("yeni_dosya.csv", sep=';',index=False)
