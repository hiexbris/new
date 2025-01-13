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

for i in range(30):
    if i < 5:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        paper['label'] = 0
        print(f"{i+1}.Done")
        training_data.append(paper)
    elif i < 9:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        if i == 5 or i == 6:
            paper['label'] = 1
        elif i == 7 or i == 8:
            paper['label'] = 2
        training_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R0{i+1}.pdf")
        if i == 9 or i == 10 or i == 21 or i == 22 or i == 23:
            paper['label'] = 3
        elif i == 11 or i == 12 or i == 24 or i == 25 or i == 26:
            paper['label'] = 4
        elif i == 13 or i == 14 or i == 27 or i == 28 or i == 29:
            paper['label'] = 5
        elif i == 15 or i == 16 or i == 17:
            paper['label'] = 1
        elif i == 18 or i == 19 or i == 20:
            paper['label'] = 2
        training_data.append(paper)
        print(f"{i+1}.Done")


import json
with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
