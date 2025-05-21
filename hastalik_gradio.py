import gradio as gr
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan Aygıt:", device)
# 1. Model ve tokenizer'ı kaydedilen klasörden yükle
model_path = "./berturk_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# 2. Etiket isimlerini JSON dosyasından yükle
with open("data/id_to_label.json", "r", encoding="utf-8-sig") as f:
    id_to_label = json.load(f)

# 3. Tahmin fonksiyonu
def hastalik_tahmin_et(belirti_metni):
    inputs = tokenizer(belirti_metni, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = int(torch.argmax(outputs.logits, dim=-1))
        tahmin = id_to_label[str(predicted_class_id)]
    return f"Tahmin edilen hastalık: {tahmin}"

# 4. Gradio arayüzü
interface = gr.Interface(
    fn=hastalik_tahmin_et,
    inputs=gr.Textbox(lines=3, label="Belirtileri giriniz"),
    outputs=gr.Textbox(label="Tahmin Edilen Hastalık"),
    title="BERTurk Hastalık Tahmin Sistemi"
)

interface.launch()
