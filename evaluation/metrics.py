import numpy as np

def ndcg_at_k(scores, k=10):
    return np.mean(scores)