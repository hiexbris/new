from zero_shot import Zero_shot

classifier = Zero_shot()

test_data = []

for i in range(135):
    if i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P00{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")
    elif i < 99:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P0{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\Test\\P{i+1}.pdf")
        test_data.append(paper)
        print(f"{i+1}.Done")


import json

with open("test_data.json", "w") as file:
    json.dump(test_data, file, indent=4) 