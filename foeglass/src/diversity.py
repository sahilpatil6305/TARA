import math
import re

def compute_similarity(str1, str2):
    # Simple word-token overlap based cosine similarity to avoid bringing in large sentence-transformer dependencies
    # since we want this lightweight CPU-compatible.
    # Alternatively we could use sklearn's TfidfVectorizer.
    from collections import Counter
    
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        words = re.compile(r'\w+').findall(text.lower())
        return Counter(words)

    vector1 = text_to_vector(str1)
    vector2 = text_to_vector(str2)
    return get_cosine(vector1, vector2)

def diversity_score(new_prompt, history_prompts):
    """
    Computes diversity score as (1 - max similarity with history prompts).
    If history is empty, returns 1.0 (maximum diversity).
    """
    if not history_prompts:
        return 1.0
        
    similarities = [compute_similarity(new_prompt, old_p) for old_p in history_prompts]
    max_sim = max(similarities)
    return 1.0 - max_sim
