"""Project V1 engineering RMSE baselines, not an exact SiMDex reproduction."""
import numpy as np
ROUTES = ('translation', 'rotation', 'fingertip', 'wrist')

def motion_distances(query, candidate, query_hands, candidate_hands):
    hands = [h for h in ('left', 'right') if query_hands[h]]
    if not hands or any(not candidate_hands[h] for h in hands):
        raise ValueError('candidate missing required query hand')
    distances = {}
    for route in ROUTES:
        values = []
        for side in hands:
            a, b = query[f'{side}_{route}'], candidate[f'{side}_{route}']
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('invalid/mismatched motion features')
            values.append(float(np.sqrt(np.mean((a-b)**2))))
        distances[route] = float(np.mean(values))
    return distances
