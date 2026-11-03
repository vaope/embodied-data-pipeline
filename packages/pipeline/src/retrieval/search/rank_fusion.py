from .motion_ranking import ROUTES

def rank_fusion(candidates):
    """Competition ranks (1,1,3 for ties); equal weights, never sum physical distances."""
    rows = [dict(c, motion_ranks={}) for c in candidates]
    for route in ROUTES:
        ordered = sorted(rows, key=lambda c: (c['motion_distances'][route], c['clip_id']))
        last, rank = None, 0
        for position, row in enumerate(ordered, 1):
            distance = row['motion_distances'][route]
            if distance != last:
                rank = position
            row['motion_ranks'][route] = rank
            last = distance
    for row in rows:
        row['rank_sum'] = sum(row['motion_ranks'].values())
    rows.sort(key=lambda r: (r['rank_sum'], -r['semantic_similarity'], r['clip_id']))
    for rank, row in enumerate(rows, 1):
        row['raw_rank'] = rank
    return rows
