from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample(sequence, start, duration, cfg):
    t, valid = sequence.timestamps_s, sequence.valid_mask
    if duration < cfg['min_duration_s']:
        raise ValueError('insufficient duration')
    inside = (t >= start-1e-9) & (t <= start+duration+1e-9)
    if inside.sum() < 3 or valid[inside].mean() < cfg['min_valid_ratio']:
        raise ValueError('low valid ratio / insufficient frames')
    vt = t[valid]
    if start < vt[0]-1e-9 or start+duration > vt[-1]+1e-9:
        raise ValueError('invalid window endpoint; no extrapolation')
    relevant = (vt[1:] > start) & (vt[:-1] < start+duration)
    if np.any(np.diff(vt)[relevant] > cfg['max_gap_s']+1e-9):
        raise ValueError('tracking/time gap exceeds limit')
    count = max(3, round(duration*cfg['resample_fps'])+1)
    grid = np.linspace(start, start+duration, count)
    p = np.stack([np.interp(grid, vt, sequence.position_m[valid, j]) for j in range(3)], axis=1)
    q = Slerp(vt, Rotation.from_quat(sequence.rotation_xyzw[valid]))(np.clip(grid, vt[0], vt[-1])).as_quat()
    return replace(sequence, timestamps_s=grid-start, position_m=p, rotation_xyzw=q,
                   valid_mask=np.ones(count, bool), grasp_signal=None, grasp_valid_mask=None,
                   metadata=dict(sequence.metadata, span_s=[float(start), float(start+duration)],
                                 source_valid_ratio=float(valid[inside].mean())))


def query_windows(sequence, duration, cfg):
    spans = []
    for ratio in cfg['duration_ratios']:
        length = duration*ratio
        last = sequence.timestamps_s[-1]-length
        if last < 0:
            continue
        starts = np.arange(0, last+1e-9, cfg['stride_s']).tolist()
        if not starts or last-starts[-1] > 1e-8:
            starts.append(float(last))
        spans.extend((float(s), float(length)) for s in starts)
    # Uniform coverage rather than truncating the start of a long episode.
    if len(spans) > cfg['max_windows']:
        spans = [spans[i] for i in np.linspace(0, len(spans)-1, cfg['max_windows']).round().astype(int)]
    return spans


def sequence_distance(a, b, cfg):
    """Scalar feature streams only. Banded DTW limits consecutive stretch steps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return None
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    if not cfg['use_dtw']:
        if len(a) != len(b):
            # Explicit bounded duration-ratio ablation: interpolate scalar streams.
            b = np.stack([np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b[:, j]) for j in range(b.shape[1])], axis=1)
        return float(np.mean(np.abs(a-b)))
    n, m = len(a), len(b)
    band = max(abs(n-m), int(np.ceil(max(n, m)*cfg['dtw_band_fraction'])))
    run = cfg['dtw_max_run']
    # State 0=diagonal, 1..run=horizontal, run+1..2run=vertical.
    cost = np.full((n+1, m+1, 1+2*run), np.inf)
    cost[0, 0, 0] = 0
    steps = np.zeros_like(cost)
    for i in range(1, n+1):
        for j in range(max(1, i-band), min(m, i+band)+1):
            local = float(np.mean(abs(a[i-1]-b[j-1])))
            choices = [(0, i-1, j-1, list(range(1+2*run))),
                       (1, i, j-1, [0]+list(range(run+1, 2*run+1))),
                       (run+1, i-1, j, list(range(run+1)))]
            choices += [(k, i, j-1, [k-1]) for k in range(2, run+1)]
            choices += [(k, i-1, j, [k-1]) for k in range(run+2, 2*run+1)]
            for state, pi, pj, previous in choices:
                best = min(previous, key=lambda k: cost[pi, pj, k])
                cost[i, j, state] = cost[pi, pj, best]+local
                steps[i, j, state] = steps[pi, pj, best]+1
    state = int(np.argmin(cost[n, m]))
    if not np.isfinite(cost[n, m, state]):
        return None
    return float(cost[n, m, state]/steps[n, m, state])
