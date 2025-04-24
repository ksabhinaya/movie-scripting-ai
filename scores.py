from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load reference script from file
with open("bbscript (1).txt", "r", encoding="utf-8") as file:
    reference_script = file.read()

# Load generated script from file
with open("movie_script (4).txt", "r", encoding="utf-8") as file:
    generated_script = file.read()

# Convert text to embeddings
reference_embedding = model.encode([reference_script])
generated_embedding = model.encode([generated_script])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(reference_embedding, generated_embedding)[0][0]

# Print the result
print(f"Cosine Similarity Score: {cosine_sim:.4f}")