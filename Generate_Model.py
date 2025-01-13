import json
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

torch.cuda.empty_cache()

def tokenize_with_sliding_window(tokenizer, texts, labels=None, max_length=512, stride=128):
    """
    Tokenize long texts using a sliding window approach.
    """
    input_ids, attention_masks, new_labels = [], [], []
    for idx, text in enumerate(texts):
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
            padding="max_length",
        )
        input_ids.extend(tokenized["input_ids"])
        attention_masks.extend(tokenized["attention_mask"])
        if labels:
            new_labels.extend([labels[idx]] * len(tokenized["input_ids"]))  # Repeat labels for all chunks
    return Dataset.from_dict({'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': new_labels})


def load_methodology_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    methodologies, labels = [], []
    for paper in data:
        methodologies.append(paper['text'])
        labels.append(paper['label'])
    return methodologies, labels


def preprocess_data(tokenizer, texts, labels, max_length=512):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset.from_dict({'input_ids': encodings['input_ids'], 
                                 'attention_mask': encodings['attention_mask'], 
                                 'labels': labels})
    return dataset


def all():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    json_path = 'training_data.json' 
    methodologies, labels = load_methodology_data(json_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(methodologies, labels, test_size=0.2, random_state=42)

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = tokenize_with_sliding_window(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_with_sliding_window(tokenizer, val_texts, val_labels)

    datasets = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./Model1',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-5,
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
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained('./Model1')
    tokenizer.save_pretrained('./Model1')

if __name__ == '__main__':
    all()
    