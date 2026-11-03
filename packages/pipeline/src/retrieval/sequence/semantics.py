"""Conservative English action/object contradiction checks; not an entailment model.

Unknown single-step captions retain semantic matching with an explicit diagnostic.
Merges require parsed evidence for every member, in order, on both sides.
"""
import re

ACTION = re.compile(r'\b(pick\s+up|take\s+off|put\s+down|set\s+down|turn\s+on|turn\s+off|switch\s+on|switch\s+off|pick|take|grab|lift|place|put|remove|move|open|close|push|pull|insert|pour)\b',re.I)
ALIASES = {'pick up':'take','pick':'take','grab':'take','lift':'take','put':'place','put down':'place',
           'set down':'place','take off':'remove','switch on':'turn on','switch off':'turn off'}


def atoms(caption):
    text = caption.lower().strip()
    actions = list(ACTION.finditer(text))
    found = []
    for i, match in enumerate(actions):
        action = ' '.join(match[0].split())
        action = ALIASES.get(action,action)
        tail = text[match.end():actions[i+1].start() if i+1<len(actions) else len(text)]
        obj = re.split(r'\b(?:from|on|onto|into|in|to|out of|off|with|toward|towards|and|then|while|using)\b|[.,;]',tail)[0]
        words = re.findall(r'[a-z]+',obj)
        words = [w for w in words if w not in ('a','an','the','up','down')]
        if not words or words[-1] in ('it','them','this','that','something'):
            return []
        noun = words[-1]
        if noun.endswith('s') and not noun.endswith(('ss','us')):
            noun = noun[:-1]
        found.append((action,noun))
    return found


def check_blocks(query_steps,candidate_steps):
    qa = [atoms(s.caption) for s in query_steps]
    ca = [atoms(s.caption) for s in candidate_steps]
    merged = len(query_steps)>1 or len(candidate_steps)>1
    known = all(qa) and all(ca)
    details = dict(query_atoms=qa,candidate_atoms=ca,rule='ordered_action_object',
                   scope='basic English templates; source/target locations are not enforced')
    if not known:
        details['status'] = 'unverified_merge' if merged else 'unparsed'
        return not merged, details
    # Collapse adjacent identical descriptions, e.g. two slices of the same pickup.
    def collapse(groups):
        out=[]
        for group in groups:
            for atom in group:
                if not out or out[-1]!=atom:
                    out.append(atom)
        return out
    passed = collapse(qa)==collapse(ca)
    details['status']='compatible' if passed else 'action_object_conflict'
    return passed,details
