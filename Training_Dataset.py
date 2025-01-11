from zero_shot import Zero_shot

classifier = Zero_shot()

training_data = []

for i in range(15):
    if i < 5:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\R00{i+1}.pdf")
        if i == 0:
            paper['label'] = "Non-Publishable"
            training_data.append(paper)
            print(f"{i+1}.Done")
        elif i == 1:
            paper['label'] = "Non-Publishable"
            training_data.append(paper)
            print(f"{i+1}.Done")
        elif i == 2:
            paper['label'] = "Non-Publishable"
            training_data.append(paper)
            print(f"{i+1}.Done")
        elif i == 3:
            paper['label'] = "Non-Publishable"
            training_data.append(paper)
            print(f"{i+1}.Done")
        elif i == 4:
            paper['label'] = "Non-Publishable"
            training_data.append(paper)
            print(f"{i+1}.Done")
    elif i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\R00{i+1}.pdf")
        paper['label'] = 'Publishable'
        training_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\R0{i+1}.pdf")
        paper['label'] = 'Publishable'
        training_data.append(paper)
        print(f"{i+1}.Done")


import json

with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
