import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For AI-generated text detection
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ---------------------------------------------
# 📘 Plagiarism Detection Functions
# ---------------------------------------------

def load_documents(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def compute_similarity(input_text, sources):
    all_docs = [input_text] + sources
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_docs)
    
    sim_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    max_score = max(sim_scores)

    return max_score > 0.7, max_score  # You can adjust threshold

# ---------------------------------------------
# 🤖 AI-Generated Text Detection Function
# ---------------------------------------------

# Load model once (global)
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()  # Class 1 = AI
    return ai_prob > 0.5, ai_prob
