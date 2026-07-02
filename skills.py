import numpy as np 

def calculate_skill_overlap(required_skills, candidate_skills):
    if not required_skills: 
        return np.zeros(len(candidate_skills))

    req_set = {skill.lower().strip() for skill in required_skills}
    
    matches = [
        len(req_set & {skill.lower().strip() for skill in cand}) / len(req_set) 
        for cand in candidate_skills
    ]
    
    return np.array(matches)