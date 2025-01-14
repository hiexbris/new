import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import shap
from sklearn.model_selection import train_test_split
import json

def load_and_preprocess_data(data, tokenizer, max_length=512):
    """
    Preprocess the dictionary-based dataset.
    """
    texts = []
    labels = []
    for paper in data:
        # Combine sections into a single text
        combined_text = (
            f"Abstract: {paper['abstract']} "
            f"Methodology: {paper['methodology']} "
            f"Results and Findings: {paper['result_and_findings']} "
            f"Conclusion: {paper['conclusion']}"
        )
        texts.append(combined_text)
        labels.append(paper['label'])
    
    # Map labels to numerical IDs
    unique_labels = list(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    label_ids = [label_mapping[label] for label in labels]
    
    # Tokenize texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': label_ids
    })
    
    return dataset, unique_labels, label_mapping, inverse_label_mapping

def train_model(train_dataset, val_dataset, model_name='allenai/scibert_scivocab_uncased', num_labels=6):
    """
    Fine-tune a transformer model for conference prediction.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir='./conference_model',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    return model, trainer

if __name__ == "__main__":
    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    with open('conference_dataset.json', 'r') as file:
        data = json.load(file)