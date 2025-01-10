from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json

data_path = 'training_data.json'

with open(data_path, 'r') as file:
    Training_Dataset = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

class SciBERTClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_classes=2):
        super(SciBERTClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return self.classifier(pooled_output)

# Initialize model
classifier_model = SciBERTClassifier(model)
classifier_model = classifier_model.to("cuda")


def custom_loss(predictions, targets, feedback_embeddings, feedback_penalty_weight=0.5):
    base_loss = F.cross_entropy(predictions, targets)
    penalty = 0.0

    for feedback_embedding in feedback_embeddings:
        if feedback_embedding is not None:
            penalty += F.mse_loss(feedback_embedding, torch.zeros_like(feedback_embedding))

    return base_loss + feedback_penalty_weight * penalty

def process_paper_sections(paper, tokenizer, model):
    embeddings = []
    feedback_embeddings = []
    for section, text in paper.items():
        if section not in ["label", "Reasons"]:
            # Tokenize with sliding window
            inputs = tokenizer(text, max_length=512, truncation=True, stride=256, return_tensors="pt", return_overflowing_tokens=True)
            inputs = {key: val.to("cuda") for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(embedding.mean(dim=0))  # Aggregate across sliding windows

            # Process feedback if available
            feedback = paper.get("Reasons", {}).get(section, None)
            if feedback:
                feedback_embedding = embedding.mean(dim=0)  # Average feedback window embeddings
            else:
                feedback_embedding = None
            feedback_embeddings.append(feedback_embedding)

    return torch.stack(embeddings), feedback_embeddings

X_train = []
y_train = []
feedback_train = []

for paper in Training_Dataset:
    section_embeddings, feedback_embeddings = process_paper_sections(paper, tokenizer, model)
    X_train.append(section_embeddings.mean(dim=0))  # Aggregate section embeddings
    feedback_train.append(feedback_embeddings)
    y_train.append(1 if paper["label"] == "Publishable" else 0)

X_train = torch.stack(X_train).to("cuda")
y_train = torch.tensor(y_train).to("cuda")

optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-5)
epochs = 10

for epoch in range(epochs):
    classifier_model.train()
    optimizer.zero_grad()

    outputs = classifier_model(X_train)
    loss = custom_loss(outputs, y_train, feedback_train)
    
    loss.backward()
    optimizer.step()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
X_test = []
y_test = []
section_names = []

for paper in Training_Dataset: 
    paper_vector = []
    for section, text in paper.items():
        if section != "label":
            inputs = tokenizer(text, max_length=512, truncation=True, stride=256, return_tensors="pt", return_overflowing_tokens=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                paper_vector.append(embedding.squeeze(0))
            section_names.append(section)
    X_test.append(torch.cat(paper_vector).mean(dim=0))  # Aggregate section embeddings
    y_test.append(1 if paper["label"] == "Publishable" else 0)


X_test = torch.stack(X_test).to("cuda")
y_test = torch.tensor(y_test).to("cuda")

classifier_model.eval()
with torch.no_grad():
    outputs = classifier_model(X_test)
    predictions = torch.argmax(outputs, dim=1)

print(predictions.cpu().numpy())