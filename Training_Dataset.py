import json
from PyPDF2 import PdfReader
import sys
sys.stdout.reconfigure(encoding='utf-8') 

def extract_text_from_pdf(pdf_path):
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

training_data = []

for i in range(15):
    if i < 5:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        paper['label'] = 0
        print(f"{i+1}.Done")
        training_data.append(paper)
    elif i < 9:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        paper['label'] = 1
        training_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R0{i+1}.pdf")
        paper['label'] = 1
        training_data.append(paper)
        print(f"{i+1}.Done")


import json
with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
