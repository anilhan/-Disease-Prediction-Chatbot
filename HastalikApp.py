import gradio as gr
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import json
import requests

# Cihaz tanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan Aygıt:", device)

# BERTurk model ve tokenizer yükleme
model_path = "./berturk_model"
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# Etiket isimlerini yükle
with open("data/id_to_label.json", "r", encoding="utf-8-sig") as f:
    id_to_label = json.load(f)

# Doğal dil açıklaması almak için Hugging Face API üzerinden GPT-2 benzeri Türkçe destekli bir modele istek at
# (örnek model: "cahya/gpt2-small-turkish")
def gpt2_aciklama_al(hastalik_adi):
    api_url = "https://api-inference.huggingface.co/models/ytu-ce-cosmos/turkish-gpt2-large"
    headers = {"Authorization": "Bearer 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"}
    prompt = f"{hastalik_adi} hastalığı hakkında kısa bir bilgilendirme yap."
    payload = {"inputs": prompt, "max_new_tokens": 200}
    resp = requests.post(api_url, headers=headers, json=payload)
    if resp.status_code==200:
        return resp.json()[0]["generated_text"].replace(prompt, "").strip()
    else:
        return f"Açıklama alınamadı: {resp.status_code}"

# Hastalık tahmini ve doğal dilli açıklama fonksiyonu
def hastalik_tahmin_et(belirti_metni):
    inputs = tokenizer(belirti_metni, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = int(torch.argmax(outputs.logits, dim=-1))
        hastalik_adi = id_to_label[str(predicted_class_id)]

    # Doğal dil yanıt al
    aciklama = gpt2_aciklama_al(hastalik_adi)
    return f"Tahmin: {hastalik_adi}\n\nAçıklama: {aciklama}"

# Gradio arayüz
interface = gr.Interface(
    fn=hastalik_tahmin_et,
    inputs=gr.Textbox(lines=3, label="Belirtileri giriniz"),
    outputs=gr.Textbox(label="Model Cevabı"),
    title="BERTurk + GPT2 Hastalık Tahmin ve Açıklama Sistemi"
)

interface.launch()
