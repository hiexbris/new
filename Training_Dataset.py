from zero_shot import Zero_shot

classifier = Zero_shot()

training_data = []

for i in range(15):
    if i < 5:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        if i == 0:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Methodology": ["The Paper doesn't compare the proposed method with other in-use methods."],
                "Results and Findings": [
                    "Limited Data, Very Few Test cases which leads to less confident claim.",
                    "Low accuracy of the tests conducted."
                    ]
                }
            paper['Reasons'] = reason_dict
            training_data.append(paper)
        elif i == 1:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Abstract": ["No experimental data to back the claims."],
                "Results and Findings": [
                    "No quantitative analysis like accuracy, error rates.",
                    "Theoretical claims, with no backing from actual data."
                ],
                "Methodology": ["No comparison with existing studies."]
            }
            paper['Reasons'] = reason_dict
            training_data.append(paper)
        elif i == 2:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Abstract": ["Lack of evidence and credible justification."],
                "Methodology": [
                    "No experiments, data, or tests; no results.",
                    "Does not describe how the method is implemented, tested, or evaluated."
                ],
                "General": ["The paper jumps between topics making it completely inappropriate, and lacks focus."]
            }
            paper['Reasons'] = reason_dict
            training_data.append(paper)
        elif i == 3:
            paper['label'] = "Non-Publishable"
            reason_dict ={
                "Abstract": ["No theoretical backing or proofs."],
                "Methodology": ["No comparison with ongoing methods of this field."],
                "Results and Findings": [
                    "Lacks experimental evidence, test data.",
                    "Lacks real-world applicability."
                ]
            }
            paper['Reasons'] = reason_dict
            training_data.append(paper)
        elif i == 4:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Methodology": [
                    "Uses existing techniques.",
                    "Lacks real-world scenario."
                ],
                "Results and Findings": [
                    "Less test cases, datasets.",
                    "No comparison with existing methods.",
                    "No statistical analysis."
                ],
                "General": ["Lack of clear explanation, repetitive text."]
            }
            paper['Reasons'] = reason_dict
            training_data.append(paper)
    elif i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        paper['label'] = 'Publishable'
        paper['Reasons'] = {}
        training_data.append(paper)
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")
        paper['label'] = 'Publishable'
        paper['Reasons'] = {}
        training_data.append(paper)


import json

with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)  # Use indent=4 for pretty formatting
