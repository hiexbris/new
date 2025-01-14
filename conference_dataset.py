import json
from PyPDF2 import PdfReader
import sys
sys.stdout.reconfigure(encoding='utf-8') 

training_data = []

def extract_text_from_pdf(pdf_path):
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

for i in range(5, 30):
    if i < 9:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        if i in [5, 6]:
            paper['label'] = 0
        elif i in [7, 8]:
            paper['label'] = 1
        training_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R0{i+1}.pdf")
        if i in [9, 10, 21, 22, 23]:
            paper['label'] = 2
        elif i in [11, 12, 24, 25, 26]:
            paper['label'] = 3
        elif i in [13, 14, 27, 28, 29]:
            paper['label'] = 4
        elif i in [15, 16, 17]:
            paper['label'] = 0
        elif i in [18, 19, 20]:
            paper['label'] = 1
        training_data.append(paper)
        print(f"{i+1}.Done")


import json
with open("conference_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
