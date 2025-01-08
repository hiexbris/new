from zero_shot import Zero_shot

classifier = Zero_shot()

classified_results = classifier.sections("D:\\KDAG Hackathon\\KDAG-Hackathon\\P001.pdf")

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import re

# Load SciBERT model and tokenizer
model_name = 'allenai/scibert_scivocab_cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_sciBERT_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

publishability_results = {}
for section_name, section_text in research_paper.items():
    publishability = check_publishability(section_name, section_text)
    publishability_results[section_name] = publishability