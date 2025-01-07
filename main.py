from zero_shot import Zero_shot

classifier = Zero_shot()
classified_results = classifier.sections("D:\\KDAG Hackathon\\KDAG-Hackathon\\P001.pdf")

categories = ["Abstract", "Methodology", "Results and Findings", "Conclusion"]

for category in categories:
    if category in classified_results:
        print(f"\n==== {category} ====\n")
        for chunk, score in classified_results[category]:
            print(f"{chunk} (Confidence: {score:.2f})")
