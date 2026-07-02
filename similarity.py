import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

candidate_embeddings = np.load("artifacts/candidate_embeddings.npy")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
tfidf_matrix = load_npz("artifacts/tfidf_matrix.npz")

def calculate_embedding_similarity(jd_embedding):
    scores = cosine_similarity(jd_embedding, candidate_embeddings).flatten()
    return np.clip(scores, 0, 1) 

def calculate_tfidf_similarity(jd_text):
    jd_vector = vectorizer.transform([jd_text])
    scores = cosine_similarity(jd_vector, tfidf_matrix).flatten()
    return np.clip(scores, 0, 1)