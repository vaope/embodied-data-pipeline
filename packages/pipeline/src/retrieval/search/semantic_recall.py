import numpy as np

def semantic_recall(query, embeddings, rows, top_n=100):
    if top_n < 1:
        raise ValueError('top_n must be positive')
    scores = np.clip(embeddings @ query, -1, 1)
    order = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i]['clip_id']))[:top_n]
    return [{'clip_id': rows[i]['clip_id'], 'semantic_similarity': float(scores[i]),
             'semantic_rank': rank+1} for rank, i in enumerate(order)]
