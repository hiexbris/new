from transformers import pipeline
from PyPDF2 import PdfReader
import nltk
import sys
sys.stdout.reconfigure(encoding='utf-8') 
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should return the GPU ID (e.g., 0)
print(torch.cuda.get_device_name(0)) 

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

categories = ["Abstract", "Methodology", "Results", "Conclusion", "References"]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def classify_sections(text, classifier, categories, batch_size=3):
    nltk.download('punkt')
    lines = nltk.tokenize.sent_tokenize(text)
    results = []

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        batch_text = " ".join(batch)  # Combine 5 lines into one batch
        
        if len(batch_text.strip()) > 20:  # Avoid very short batches
            classification = classifier(batch_text, candidate_labels=categories, multi_label=True)
            label = classification['labels'][0]  # Top category
            score = classification['scores'][0]  # Confidence score
            results.append((batch_text, label, score))
    return results

pdf_path = "D:\\KDAG Hackathon\\KDAG-Hackathon\\P001.pdf"
text = extract_text_from_pdf(pdf_path)
classified_results = classify_sections(text, classifier, categories)

# for paragraph, label, score in classified_results:
#     print(f"\nParagraph:\n{paragraph}\n\nPredicted Category: {label} (Confidence: {score:.2f})")

for category, items in classified_results.items():
    print(f"\n==== {category.upper()} ====\n")
    for chunk, score in items:
        print(f"{chunk} (Confidence: {score:.2f})")