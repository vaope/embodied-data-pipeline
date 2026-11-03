"""Single-anchor recall and bounded monotonic block alignment. No learned parameters."""
from dataclasses import asdict, dataclass
from collections import defaultdict
import numpy as np
from .models import Config, validate_steps
from .semantics import check_blocks


def block_text(steps):
    return ' Then '.join(s.caption.rstrip('. ') for s in steps) if len(steps) > 1 else steps[0].caption


class Scores:
    def __init__(self, encoder):
        self.encoder = encoder
        self.vectors = {}

    def prepare(self, texts):
        missing = sorted(set(texts) - self.vectors.keys())
        if not missing:
            return
        v = np.asarray(self.encoder.encode(missing), dtype=np.float32)
        if v.ndim != 2 or len(v) != len(missing) or not np.isfinite(v).all():
            raise ValueError('invalid encoder output')
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError('zero embedding')
        self.vectors.update(zip(missing, v / norms))

    def sim(self, a, b):
        return float(np.clip(self.vectors[a] @ self.vectors[b], -1, 1))


@dataclass
class PathState:
    value: float
    mappings: tuple = ()
    skipped: tuple = ()
    missing: tuple = ()


def align(query, candidates, anchor, scores, cfg):
    """Best path per missing-count; free candidate prefixes/suffixes, mandatory key."""
    q, c = query.steps, candidates
    m, n = len(q), len(c)
    key = query.key_index
    transitions = [(1, 1), (1, 2), (2, 1)] if cfg.allow_merges else [(1, 1)]
    matches = {}
    for i in range(m):
        for j in range(n):
            for a, b in transitions:
                if i+a > m or j+b > n:
                    continue
                has_key, has_anchor = i <= key < i+a, j <= anchor < j+b
                if has_key != has_anchor:
                    continue
                compatible, semantic_check = check_blocks(q[i:i+a],c[j:j+b])
                if not compatible:
                    continue
                if any(c[t].start_s-c[t-1].end_s > cfg.max_step_gap_s for t in range(j+1, j+b)):
                    continue
                qt, ct = block_text(q[i:i+a]), block_text(c[j:j+b])
                pair = scores.sim(qt, ct)
                per_query = [scores.sim(s.caption, ct) for s in q[i:i+a]]
                per_candidate = [scores.sim(qt, s.caption) for s in c[j:j+b]]
                if min([pair] + per_query + per_candidate) < cfg.min_similarity:
                    continue
                # Cap individual contributions by the ordered group similarity.
                contributions = [min(pair, v) for v in per_query]
                matches[i,j,a,b] = (sum(contributions) - cfg.merge_penalty*(a+b-2), {
                    'query_ids': [s.step_id for s in q[i:i+a]],
                    'candidate_ids': [s.step_id for s in c[j:j+b]],
                    'query_captions': [s.caption for s in q[i:i+a]],
                    'candidate_captions': [s.caption for s in c[j:j+b]],
                    'similarity': pair, 'per_query_similarity': per_query,
                    'contributions': contributions, 'is_key': has_key,
                    'semantic_check': semantic_check,
                    'query_span_s': [q[i].start_s, q[i+a-1].end_s],
                    'candidate_span_s': [c[j].start_s, c[j+b-1].end_s]})
    results = []
    for start in range(anchor+1):
        # Buckets topologically ordered by consumed query+candidate count.
        states = {(0, start, False, 0, 0): PathState(0)}
        buckets = defaultdict(set)
        buckets[start].add((0, start, False, 0, 0))

        def put(k, p):
            if k not in states or p.value > states[k].value + 1e-12:
                states[k] = p
                buckets[k[0]+k[1]].add(k)

        for layer in range(start, m+n+1):
            for state in sorted(buckets[layer]):
                i, j, seen, gaps, missing = state
                p = states[state]
                if i == m:
                    if seen and p.mappings and (not p.skipped or p.skipped[-1] != j-1):
                        end = j-1
                        span = c[end].end_s-c[start].start_s
                        used = {x for mapping in p.mappings for x in mapping['candidate_ids']}
                        background = max(0., span-sum(s.end_s-s.start_s for s in c[start:j] if s.step_id in used))
                        results.append(dict(source_id=c[start].source_id,
                            start_s=c[start].start_s, end_s=c[end].end_s,
                            video_path=c[start].video_path, anchor_step_id=c[anchor].step_id,
                            status='complete' if missing == 0 else 'partial',
                            score=p.value/m, coverage=(m-missing)/m,
                            weakest_similarity=min(v for x in p.mappings for v in x['contributions']),
                            mappings=list(p.mappings), missing_query_ids=[q[t].step_id for t in p.missing],
                            skipped_candidate_ids=[c[t].step_id for t in p.skipped],
                            unmatched_seconds=background,
                            steps=[asdict(s) for s in c[start:j]]))
                    continue
                # Missing required steps remain explicit diagnostic paths; key cannot drop.
                if i != key:
                    put((i+1,j,seen,gaps,missing+1), PathState(p.value,p.mappings,p.skipped,p.missing+(i,)))
                if j == n:
                    continue
                interval_gap = max(0., c[j].start_s-c[j-1].end_s) if j > start else 0.
                if interval_gap > cfg.max_step_gap_s:
                    continue
                if c[j].end_s-c[start].start_s > cfg.max_span_s:
                    continue
                if p.mappings and j != anchor and gaps < cfg.max_skips:
                    cost = cfg.skip_penalty + cfg.gap_penalty_per_s*(interval_gap+c[j].end_s-c[j].start_s)
                    put((i,j+1,seen,gaps+1,missing), PathState(p.value-cost,p.mappings,p.skipped+(j,),p.missing))
                for a,b in transitions:
                    match = matches.get((i,j,a,b))
                    if match is None or c[j+b-1].end_s-c[start].start_s > cfg.max_span_s:
                        continue
                    gain, mapping = match
                    internal_gap = sum(max(0.,c[t].start_s-c[t-1].end_s) for t in range(j+1,j+b))
                    value = p.value+gain-cfg.gap_penalty_per_s*(interval_gap+internal_gap)
                    put((i+a,j+b,seen or mapping['is_key'],gaps,missing),
                        PathState(value,p.mappings+(mapping,),p.skipped,p.missing))
    return results


def rank_key(r):
    return (-r.get('coverage',1), -r['score'], r['end_s']-r['start_s'], r['source_id'], r['start_s'])


def suppress(rows, limit, iou):
    chosen = []
    for r in sorted(rows, key=rank_key):
        duplicate = False
        for other in chosen:
            if r['source_id'] != other['source_id']:
                continue
            intersection = max(0., min(r['end_s'],other['end_s'])-max(r['start_s'],other['start_s']))
            union = max(r['end_s'],other['end_s'])-min(r['start_s'],other['start_s'])
            if intersection/union >= iou:
                duplicate = True
                break
        if not duplicate:
            chosen.append(r)
        if len(chosen) >= limit:
            break
    return chosen


def search(query, steps, encoder, cfg=None):
    cfg = cfg or Config()
    steps = sorted(steps, key=lambda s:(s.source_id,s.start_s,s.step_id))
    validate_steps(steps)
    grouped = defaultdict(list)
    for s in steps:
        grouped[s.source_id].append(s)
    scores = Scores(encoder)
    key_text = query.steps[query.key_index].caption
    scores.prepare([key_text]+[s.caption for s in steps])
    anchors = sorted(((scores.sim(key_text,s.caption),s) for s in steps), key=lambda x:(-x[0],x[1].step_id))
    selected = [(v,s) for v,s in anchors if v >= cfg.anchor_min_similarity][:cfg.anchor_top_k]
    windows = []
    texts = [s.caption for s in query.steps]
    texts += [block_text(query.steps[i:i+2]) for i in range(len(query.steps)-1)]
    for value, anchor in selected:
        chain = grouped[anchor.source_id]
        idx = chain.index(anchor)
        left = max(0,idx-cfg.neighbor_steps)
        right = min(len(chain),idx+cfg.neighbor_steps+1)
        while left < idx and anchor.start_s-chain[left].end_s > cfg.neighbor_seconds:
            left += 1
        while right > idx+1 and chain[right-1].start_s-anchor.end_s > cfg.neighbor_seconds:
            right -= 1
        c = chain[left:right]
        texts += [s.caption for s in c]+[block_text(c[i:i+2]) for i in range(len(c)-1)]
        windows.append((c,idx-left))
    # Independent all-corpus caption-concatenation baseline, no anchor restriction.
    concat_blocks = []
    full_query = block_text(query.steps)
    texts.append(full_query)
    for chain in grouped.values():
        if not cfg.concatenated_baseline:
            break
        for i in range(len(chain)):
            for length in range(1,min(2*len(query.steps),len(chain)-i)+1):
                block = chain[i:i+length]
                if block[-1].end_s-block[0].start_s > cfg.max_span_s:
                    break
                if any(block[t].start_s-block[t-1].end_s > cfg.max_step_gap_s for t in range(1,length)):
                    break
                concat_blocks.append(block)
                texts.append(block_text(block))
    scores.prepare(texts)
    candidates = [r for c,a in windows for r in align(query,c,a,scores,cfg)]
    complete = suppress([r for r in candidates if r['status']=='complete'],cfg.top_k,cfg.nms_iou)
    partial = suppress([r for r in candidates if r['status']=='partial'],cfg.top_k,cfg.nms_iou)
    concat = [dict(source_id=b[0].source_id,start_s=b[0].start_s,end_s=b[-1].end_s,
                   step_ids=[s.step_id for s in b],score=scores.sim(full_query,block_text(b))) for b in concat_blocks]
    pools = {}
    if cfg.rerank_pool_size:
        pools = dict(complete=suppress([r for r in candidates if r['status']=='complete'],cfg.rerank_pool_size,cfg.nms_iou),
            single_key=[dict(step_id=s.step_id,source_id=s.source_id,start_s=s.start_s,end_s=s.end_s,score=v)
                        for v,s in anchors[:cfg.rerank_pool_size]])
    return dict(schema_version='sequence.v1', query=asdict(query), config=asdict(cfg), candidate_pools=pools,
        anchors=[dict(step_id=s.step_id,source_id=s.source_id,similarity=v) for v,s in selected],
        complete=complete,partial=partial,
        baselines=dict(single_key=[dict(step_id=s.step_id,source_id=s.source_id,start_s=s.start_s,end_s=s.end_s,score=v)
                                   for v,s in anchors[:cfg.top_k]],
                       concatenated=suppress(concat,cfg.top_k,cfg.nms_iou)),
        stats=dict(corpus_steps=len(steps),corpus_sources=len(grouped),anchors=len(selected),
                   encoded_texts=len(scores.vectors),alignment_paths=len(candidates)),
        limitations=['Algorithmic complete is not human semantic acceptance.',
                     'Caption-only; basic English action/object guard, not general entailment. Locations and outcomes are not verified.',
                     'Anchor and neighborhood limits may miss globally better chains.',
                     'Thresholds are development defaults, not calibrated accuracy guarantees.'])
