from transformers import pipeline
from PyPDF2 import PdfReader
import nltk
import sys
sys.stdout.reconfigure(encoding='utf-8') 

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

categories = ["Abstract", "Methodology", "Results and Findings", "Conclusion"]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def classify_sections(text, classifier, categories, batch_size=1):
    nltk.download('punkt')
    lines = nltk.tokenize.sent_tokenize(text)
    results = {category: [] for category in categories}
    

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        batch_text = " ".join(batch) 
        
        classification = classifier(batch_text, candidate_labels=categories, multi_label=True)
        label = classification['labels'][0]  
        score = classification['scores'][0] 
        if score > 0.4:
            results[label].append((batch_text, score))          
    return results

pdf_path = "D:\\KDAG Hackathon\\KDAG-Hackathon\\P001.pdf"
text = extract_text_from_pdf(pdf_path)
classified_results = classify_sections(text, classifier, categories)

for category in categories:
    if category in classified_results:
        print(f"\n==== {category} ====\n")
        for chunk, score in classified_results[category]:
            print(f"{chunk} (Confidence: {score:.2f})")