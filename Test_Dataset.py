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

test_data = []

for i in range(135):
    if i < 9:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P00{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")
    elif i < 99:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P0{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = {}
        paper['text'] = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")


import json

with open("test_data.json", "w") as file:
    json.dump(test_data, file, indent=4) 