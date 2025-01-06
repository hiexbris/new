from PyPDF2 import PdfReader

papers = []
for i in range(10):
    if i < 9:
        reader = PdfReader(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
    else: 
        reader = PdfReader(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    papers.append(text)


from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
inputs = tokenizer(papers, padding=True, truncation=True, return_tensors="pt")
labels_tensor = torch.tensor(labels)

from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels_tensor)
dataloader = DataLoader(dataset, batch_size=2)

optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()

for epoch in range(3):  
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

model.eval()  

for j in range(5):
    reader = PdfReader(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P01{j+1}.pdf")
    test_text = ''
    for page in reader.pages:
        test_text += page.extract_text()

    inputs_test = tokenizer(test_text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs_test)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=1)
    print("Prediction:", "Valid" if prediction.item() == 1 else "Invalid") 
    
    if prediction.item() == 0:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain.llms import OpenAI

        # Define the template for the explanation
        prompt_template = """
        Given the following content from a research paper, provide a short 10-word explanation of why it is not publishable.

        Content: {paper_content}
        Explanation:
        """

        # Create the prompt template and chain
        prompt = PromptTemplate(input_variables=["paper_content"], template=prompt_template)
        llm = OpenAI(temperature=0.5)

        chain = LLMChain(prompt=prompt, llm=llm)

        # Example content from the paper (this would typically come from your data extraction step)
        paper_content = "The methodology in this paper uses an outdated technique that is not suitable for the problem being addressed."

        # Generate explanation
        explanation = chain.run({"paper_content": paper_content})
        print(explanation)


