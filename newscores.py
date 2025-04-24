import os
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from bert_score import score as bert_score
from gensim.models import KeyedVectors
import language_tool_python
from tabulate import tabulate

# Load reference and generated scripts from text files
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()

# File paths (Update these with your actual filenames)
reference_script = load_text("bbscript (1).txt")
generated_script = load_text("movie_script (4).txt")

# Tokenize text
reference_tokens = reference_script.split()
generated_tokens = generated_script.split()

# ---------------------- Perplexity (PPL) Calculation ----------------------
def calculate_perplexity(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

ppl_score = calculate_perplexity(generated_script)

# ---------------------- BLEU & Self-BLEU Calculation ----------------------
bleu_score = sentence_bleu(generated_script, [reference_script]).score / 100
self_bleu_score = sentence_bleu(generated_script, [generated_script]).score / 100  # Self-BLEU

# ---------------------- ROUGE Score Calculation ----------------------
scorer = rouge_scorer.RougeScorer(["rouge-1", "rouge-2", "rouge-l"], use_stemmer=True)
rouge_scores = scorer.score(reference_script, generated_script)

rouge_1 = rouge_scores["rouge-1"].fmeasure
rouge_2 = rouge_scores["rouge-2"].fmeasure
rouge_l = rouge_scores["rouge-l"].fmeasure

# ---------------------- BERTScore Calculation ----------------------
P, R, F1 = bert_score([generated_script], [reference_script], lang="en", rescale_with_baseline=True)
bert_f1 = F1.mean().item()

# ---------------------- Word Mover’s Distance (WMD) Calculation ----------------------
w2v_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

def calculate_wmd(ref, gen):
    ref_tokens = ref.lower().split()
    gen_tokens = gen.lower().split()
    return w2v_model.wmdistance(ref_tokens, gen_tokens)

try:
    wmd_score = calculate_wmd(reference_script, generated_script)
except Exception:
    wmd_score = float("nan")  # Assign NaN if WMD fails due to word mismatches

# ---------------------- Grammar Error Calculation ----------------------
tool = language_tool_python.LanguageTool("en-US")
grammar_errors = len(tool.check(generated_script))

# ---------------------- Store Results in Table ----------------------
results = [
    ["Perplexity (PPL)", f"{ppl_score:.4f}"],
    ["BLEU Score", f"{bleu_score:.4f}"],
    ["Self-BLEU Score", f"{self_bleu_score:.4f}"],
    ["ROUGE-1 Score", f"{rouge_1:.4f}"],
    ["ROUGE-2 Score", f"{rouge_2:.4f}"],
    ["ROUGE-L Score", f"{rouge_l:.4f}"],
    ["BERT Score", f"{bert_f1:.4f}"],
    ["Word Mover’s Distance (WMD)", f"{wmd_score:.4f}"],
    ["Grammar Errors", grammar_errors],
]

# Display Results
print(tabulate(results, headers=["Metric", "Score"], tablefmt="grid"))