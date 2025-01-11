from zero_shot import Zero_shot

classifier = Zero_shot()

test_data = []

for i in range(150):
    if i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        test_data.append(paper)
    elif i < 99:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")
        test_data.append(paper)
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P{i+1}.pdf")
        test_data.append(paper)


import json

with open("test_data.json", "w") as file:
    json.dump(test_data, file, indent=4) 