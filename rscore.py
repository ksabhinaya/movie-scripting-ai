from rouge_score import rouge_scorer

 #Load reference script from file
with open("bbscript (1).txt", "r", encoding="utf-8") as file:
    reference_script = file.read()

# Load generated script from file
with open("movie_script (4).txt", "r", encoding="utf-8") as file:
    generated_script = file.read()


# Initialize ROUGE scorer (Corrected ROUGE types)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_script, generated_script)

# Display scores
print(f"ROUGE-1 Score: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Score: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Score: {scores['rougeL'].fmeasure:.4f}")