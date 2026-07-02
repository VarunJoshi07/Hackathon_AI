import numpy as np

def calculate_credibility(df):
    score = np.ones(len(df))
    
    score -= (df.get('has_duplicate_skills', 0) * 0.10)
    score -= (df.get('is_incomplete', 0) * 0.20)
    score -= (df.get('is_missing_experience', 0) * 0.20)
    
    return np.clip(score, 0, 1)