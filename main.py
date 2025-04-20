from utils import load_documents, preprocess, compute_similarity, detect_ai_text

# Define file paths
input_doc = 'uploads/input.txt'
source_docs = ['data/source1.txt', 'data/source2.txt']

# Load and preprocess documents
input_text = preprocess(load_documents(input_doc))
sources = [preprocess(load_documents(doc)) for doc in source_docs]

# --- Plagiarism Detection ---
plagiarized, score = compute_similarity(input_text, sources)
print(f"Plagiarism Detected: {plagiarized}")
print(f"Max Similarity Score: {score:.2f}")

# --- AI-Generated Content Detection ---
ai_detected, ai_score = detect_ai_text(input_text)
print(f"AI-Generated Text: {ai_detected}")
print(f"AI-Likeness Score: {ai_score:.2f}")
