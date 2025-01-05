import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()
# features = vectorizer.fit_transform(papers)

# print(features.toarray())  

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text += page.get_text()  
    
    return text

pdf_path = "P001.pdf"
paper_text = extract_text_from_pdf(pdf_path)
print(paper_text)  
