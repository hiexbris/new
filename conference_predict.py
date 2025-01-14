from sympy import Predicate
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def predict_methodology(methodology_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained('./Con').to(device)
    tokenizer = AutoTokenizer.from_pretrained('./Con')
    
    tokenized = tokenizer(
        methodology_text,
        max_length=512,
        truncation=True,
        return_overflowing_tokens=True,
        stride=128,
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    all_logits = []

    with torch.no_grad():
        for i in range(input_ids.size(0)):  # Loop through each chunk
            outputs = model(input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
            all_logits.append(outputs.logits)

    aggregated_logits = torch.mean(torch.stack(all_logits), dim=0)
    prediction = torch.argmax(aggregated_logits).item()

    return prediction


import json

with open("training_data.json", 'r') as file:
    data = json.load(file)
for paper in data:
    prediction = predict_methodology(paper['text'])
    if prediction == 0:
        print('CERN')
    elif prediction == 1:
        print('EMNLP')
    elif prediction == 2:
        print('KDD')
    elif prediction == 3:
        print('NeurIPS')
    elif prediction == 4:
        print('TMLR')
    