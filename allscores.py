import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load real movie script as reference
with open("bbscript (1).txt", "r", encoding="utf-8") as file:
    reference_script = file.read().lower()

# Load generated script
with open("movie_script (4).txt", "r", encoding="utf-8") as file:
    generated_script = file.read().lower()

# Tokenization using split() instead of word_tokenize()
reference_tokens = reference_script.split()
generated_tokens = generated_script.split()

# BLEU Score Calculation
bleu_score = sentence_bleu(
    [reference_tokens], generated_tokens, 
    smoothing_function=SmoothingFunction().method1
)

# ROUGE Score Calculation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference_script, generated_script)

# Cosine Similarity Calculation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([reference_script, generated_script])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Print Evaluation Metrics
print(f"BLEU Score: {bleu_score:.4f}")
print(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")
print(f"Cosine Similarity Score: {cosine_sim:.4f}")