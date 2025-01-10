from zero_shot import Zero_shot

classifier = Zero_shot()

for i in range(15):
    if i < 5:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        
    elif i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")