from transformers import pipeline
from PyPDF2 import PdfReader
import nltk
import sys
sys.stdout.reconfigure(encoding='utf-8') 

class Zero_shot():

    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

    def sections(self, pdf_path):

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
                if score > 0.2:
                    results[label].append((batch_text, score))          
            return results

        text = extract_text_from_pdf(pdf_path)
        classified_results = classify_sections(text, self.classifier, categories)

        return classified_results

        for category in categories:
            if category in classified_results:
                print(f"\n==== {category} ====\n")
                for chunk, score in classified_results[category]:
                    print(f"{chunk} (Confidence: {score:.2f})")