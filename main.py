from zero_shot import Zero_shot

classifier = Zero_shot()

for i in range(1):
    if i < 5:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        if i == 0:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Methodology": ["The Paper doesn't compare the proposed method with other in-use methods."],
                "Results": [
                    "Limited Data, Very Few Test cases which leads to less confident claim.",
                    "Low accuracy of the tests conducted."
                    ]
                }
            paper['reasons'] = reason_dict
            print(paper)
        elif i == 1:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Abstract": ["No experimental data to back the claims."],
                "Results": [
                    "No quantitative analysis like accuracy, error rates.",
                    "Theoretical claims, with no backing from actual data."
                ],
                "Methodology": ["No comparison with existing studies."]
            }
            paper['reasons'] = reason_dict
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
            paper['reasons'] = reason_dict
        elif i == 3:
            paper['label'] = "Non-Publishable"
            reason_dict ={
                "Abstract": ["No theoretical backing or proofs."],
                "Methodology": ["No comparison with ongoing methods of this field."],
                "Results": [
                    "Lacks experimental evidence, test data.",
                    "Lacks real-world applicability."
                ]
            }
            paper['reasons'] = reason_dict
        elif i == 4:
            paper['label'] = "Non-Publishable"
            reason_dict = {
                "Methodology": [
                    "Uses existing techniques.",
                    "Lacks real-world scenario."
                ],
                "Results": [
                    "Less test cases, datasets.",
                    "No comparison with existing methods.",
                    "No statistical analysis."
                ],
                "General": ["Lack of clear explanation, repetitive text."]
            }
            paper['reasons'] = reason_dict
    elif i < 9:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P00{i+1}.pdf")
        paper['label'] = 'Publishable'
        paper['reasons'] = {}
    else:
        paper = classifier.sections(f"D:\\KDAG Hackathon\\KDAG-Hackathon\\P0{i+1}.pdf")
        paper['label'] = 'Publishable'
        paper['reasons'] = {}