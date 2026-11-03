"""Independent temporal scoring. GT never enters generation."""
from .core import validate_segments


def temporal_score(predicted, reference, duration, thresholds=(.3, .5, .7)):
    validate_segments(predicted, duration)
    validate_segments(reference, duration)
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    iou = np.zeros((len(predicted), len(reference)))
    for i, p in enumerate(predicted):
        for j, g in enumerate(reference):
            intersection = max(0., min(p['end_s'], g['end_s']) - max(p['start_s'], g['start_s']))
            union = max(p['end_s'], g['end_s']) - min(p['start_s'], g['start_s'])
            iou[i,j] = intersection / union
    scores = {}
    for t in thresholds:
        if not 0 < t <= 1:
            raise ValueError('IOU_THRESHOLD_INVALID')
        eligible = iou >= t
        weight = eligible.astype(float) * (min(iou.shape) + 1) + np.where(eligible, iou, 0.)
        rows, cols = linear_sum_assignment(-weight)
        tp = sum(bool(eligible[r,c]) for r,c in zip(rows,cols))
        precision, recall = tp / len(predicted), tp / len(reference)
        scores[str(t)] = dict(tp=tp, fp=len(predicted)-tp, fn=len(reference)-tp,
                             precision=precision, recall=recall,
                             f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    return dict(predicted_count=len(predicted), reference_count=len(reference), thresholds=scores,
                caption_accuracy=None, e2e_joint_f1=None,
                limitation='Temporal only; caption/joint metrics require independent human judgements.')
