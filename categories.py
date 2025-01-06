from transformers import pipeline
from PyPDF2 import PdfReader

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

categories = ["Abstract", "Methodology", "Results", "Conclusion", "References"]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def classify_sections(text, classifier, categories):
    paragraphs = text.split("\n\n")
    results = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) > 20:  # Avoid classifying very short lines
            classification = classifier(paragraph, candidate_labels=categories)
            label = classification['labels'][0]  # Top prediction
            score = classification['scores'][0]  # Confidence score
            results.append((paragraph, label, score))
    return results

pdf_path = "D:\\KDAG Hackathon\\KDAG-Hackathon\\P001.pdf"
text = extract_text_from_pdf(pdf_path)
classified_results = classify_sections(text, classifier, categories)


for paragraph, label, score in classified_results:
    print(f"\nParagraph:\n{paragraph}\n\nPredicted Category: {label} (Confidence: {score:.2f})")