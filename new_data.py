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

for i in range(15):
    if i < 5:
        paper = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        dict = {'text': paper, 'label': 0}
        test_data.append(dict)
    elif i < 9:
        paper = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        dict = {'text': paper, 'label': 1}
        test_data.append(dict)
    elif i < 99:
        paper = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")
        dict = {'text': paper, 'label': 1}
        test_data.append(dict)
    else:
        paper = extract_text_from_pdf(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P{i+1}.pdf")
        test_data.append(paper)


import json

with open("test_data.json", "w") as file:
    json.dump(test_data, file, indent=4) 