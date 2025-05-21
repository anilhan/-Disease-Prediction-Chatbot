from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd
import evaluate
from sklearn.model_selection import train_test_split

# Veri yukleme
df = pd.read_csv("data/onislenmis_bert_turk_veriseti.csv", sep=";", encoding='utf-8-sig')

# Eğitim / validation / test ayrımı
df = df.dropna(subset=["Belirti", "label"])
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["Belirti"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Tokenizer ve model
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=30)

# Tokenization
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()}).map(tokenize_function, batched=True)
val_dataset = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()}).map(tokenize_function, batched=True)

# Değerlendirme metriği
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./berturk_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("./berturk_model")
tokenizer.save_pretrained("./berturk_model")