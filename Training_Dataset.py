from zero_shot import Zero_shot

classifier = Zero_shot()

training_data = []

for i in range(15):
    if i < 5:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        paper['label'] = 0
        print(f"{i+1}.Done")
        training_data.append(paper)
    elif i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R00{i+1}.pdf")
        if i == 5 or i == 6:
            paper['label'] = 1
        elif i == 7 or i == 8:
            paper['label'] = 2
        training_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Train\\R0{i+1}.pdf")
        if i == 9 or i == 10:
            paper['label'] = 3
        elif i == 11 or i == 12:
            paper['label'] = 4
        elif i == 13 or i == 14:
            paper['label'] = 5
        training_data.append(paper)
        print(f"{i+1}.Done")


import json
with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
