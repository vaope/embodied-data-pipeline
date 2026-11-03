"""Robot-to-Ego motion reranking within a frozen semantic pool.

No training. Axis-invariant observed-effector features participate in scoring;
unmapped directed wrist/EE trajectories are diagnostics, never claimed aligned.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math
import numpy as np
from ..adapters.droid_motion import DROIDAdapter
from ..adapters.egodex_wrist_motion import EgoWristAdapter
from ..search.event_behavior import crop_event, features, choose_active_hand, behavior_distance
from ..search.event_local import local_config, build_event_representation
from .engine import suppress


@dataclass(frozen=True)
class MotionConfig:
    weight: float = .03
    nominal_robot_fps: float = 15.
    min_valid_ratio: float = .8
    max_gap_s: float = .15

    def __post_init__(self):
        if not math.isfinite(self.weight) or not 0<=self.weight<=.1:
            raise ValueError('motion weight must be finite in [0,.1]')
        if not math.isfinite(self.nominal_robot_fps) or self.nominal_robot_fps<=0:
            raise ValueError('invalid robot fps')
        if not 0<self.min_valid_ratio<=1 or not math.isfinite(self.max_gap_s) or self.max_gap_s<=0:
            raise ValueError('invalid motion validity configuration')


def combine(semantic, pairs, required_steps, weight):
    """Each query step votes once. Missing evidence has zero adjustment."""
    used=sum(p['query_count'] for p in pairs if p['status']=='ok')
    adjustment=sum(p['query_count']*(1/(1+p['distance'])-.5)
                   for p in pairs if p['status']=='ok')/required_steps*weight
    return dict(semantic_score=float(semantic),score=float(semantic+adjustment),
        motion_adjustment=float(adjustment),motion_coverage=used/required_steps,
        motion_status='ok' if used==required_steps else 'partial' if used else 'unavailable',
        motion_pairs=pairs)


class MotionReranker:
    def __init__(self, robot_csv, manifest_rows, config=None):
        self.config=config or MotionConfig()
        self.rows={r.get('clip_id',r.get('step_id')):r for r in manifest_rows}
        self.robot_csv=Path(robot_csv).resolve()
        self.robot=DROIDAdapter().read(self.robot_csv,nominal_fps=self.config.nominal_robot_fps)
        self.cfg=dict(resample_fps=15.,min_valid_ratio=self.config.min_valid_ratio,
            max_gap_s=self.config.max_gap_s,min_duration_s=.1,confidence_threshold=.2,
            rotation_tolerance=.001,stationary_epsilon_m=.00001)
        self.local_cfg=local_config()
        self.cache={};self.failures={}

    def describe(self, seq, feature, hand):
        out=dict(hand=hand,valid_ratio=feature['valid_ratio'],duration_s=float(seq.timestamps_s[-1]),
            path_length_m=float(feature['total_path_length_m']),activity=float(feature['activity']),
            pose_sha256=hashlib.sha256(seq.position_m.tobytes()+seq.rotation_xyzw.tobytes()+seq.valid_mask.tobytes()).hexdigest())
        try:
            t=build_event_representation(seq,self.local_cfg,side=hand)
            ids=np.linspace(0,len(t.positions_fixed)-1,17).round().astype(int)
            out['trajectory_fixed_local_m']=t.positions_fixed[ids].tolist()
            out['mapping_declared']=False
        except ValueError as exc:
            out['trajectory_diagnostic_error']=str(exc)
        return out

    def query_block(self, query, ids):
        steps=[s for s in query.steps if s.step_id in ids]
        if len(steps)!=len(ids):raise ValueError('unknown query motion step')
        fps=self.config.nominal_robot_fps
        start=round(min(s.start_s for s in steps)*fps)
        end=round(max(s.end_s for s in steps)*fps)
        key=('robot',start,end)
        if key not in self.cache:
            seq=crop_event(self.robot,start,end)
            feat=features(seq,self.cfg)
            feat['activity']=float(np.mean(feat['speed'])+.05*np.mean(feat['omega']))
            self.cache[key]=(feat,self.describe(seq,feat,'robot_ee'))
        return self.cache[key]

    def ego_block(self, ids):
        rows=[self.rows[i] for i in ids]
        paths={r['hdf5_path'] for r in rows};fpss={r['fps'] for r in rows}
        if len(paths)!=1 or len(fpss)!=1:raise ValueError('motion block crosses sources or fps')
        start=min(r['start_frame'] for r in rows);end=max(r['end_frame'] for r in rows)
        key=('ego',next(iter(paths)),start,end)
        if key in self.failures:raise ValueError(self.failures[key])
        if key not in self.cache:
            row=dict(rows[0],start_frame=start,end_frame=end,start_time=start/rows[0]['fps'],end_time=end/rows[0]['fps'])
            try:
                hands,issues=EgoWristAdapter().read(row,grasp=None)
                hand,feat,activities,errors=choose_active_hand(hands,self.cfg)
                self.cache[key]=(feat,dict(self.describe(hands[hand],feat,hand),hand_activities=activities,issues=dict(issues,**errors)))
            except (ValueError,KeyError,OSError) as exc:
                self.failures[key]=str(exc);raise
        return self.cache[key]

    def score(self, query, row, single=False):
        maps=[dict(query_ids=[query.key_step_id],candidate_ids=[row['step_id']])] if single else row['mappings']
        pairs=[]
        for m in maps:
            pair=dict(query_ids=m['query_ids'],candidate_ids=m['candidate_ids'],query_count=len(m['query_ids']),status='unavailable')
            try:
                qf,qd=self.query_block(query,m['query_ids']);ef,ed=self.ego_block(m['candidate_ids'])
                distance,parts=behavior_distance(qf,ef)
                if not math.isfinite(distance) or distance<0:raise ValueError('invalid motion distance')
                pair.update(status='ok',distance=distance,parts=parts,robot=qd,ego=ed)
                if qd['activity']<.01 or ed['activity']<.01:pair['warning']='low activity; may not identify the operating hand'
            except (ValueError,KeyError,OSError,IndexError) as exc:
                pair['reason']=str(exc)
            pairs.append(pair)
        return dict(row,**combine(row['score'],pairs,1 if single else len(query.steps),self.config.weight))

    def apply(self, result, query, cfg):
        if not result.get('candidate_pools'):raise ValueError('motion reranking requires rerank_pool_size > 0')
        result['semantic_complete']=result['complete']
        result['semantic_partial']=result['partial']
        scored={}
        for kind in ('complete','single_key'):
            pool=result['candidate_pools'][kind]
            rows=[dict(self.score(query,r,single=kind=='single_key'),semantic_rank=i) for i,r in enumerate(pool,1)]
            rows.sort(key=lambda r:(-r['score'],r['semantic_rank']))
            for i,r in enumerate(rows,1):r['fusion_rank']=i
            scored[kind]=rows
        result['motion_pools']=scored
        result['complete']=suppress(scored['complete'],cfg.top_k,cfg.nms_iou)
        result['baselines']['single_key_motion']=scored['single_key'][:cfg.top_k]
        result['motion_provenance']=dict(route='robot_to_ego',feature='effector-scalar-profile-v1',
            config=asdict(self.config),robot_csv=str(self.robot_csv),robot_csv_sha256=hashlib.sha256(self.robot_csv.read_bytes()).hexdigest(),
            formula='F = semantic + weight * sum_over_valid_blocks(query_count * (1/(1+d)-0.5))/required_steps',
            missing='neutral adjustment; explicit status/reason, not a zero distance',
            directed_xyz='diagnostic only; wrist/EE axis mapping is undeclared',
            hand_selection='query-independent maximum activity among valid Ego hands; heuristic',
            timing=self.robot.metadata['timestamp_source'],cache_blocks=len(self.cache))
        result['limitations']=[s.replace('Caption-only;', 'Caption-based semantic recall; observed-effector motion reranking;') for s in result.get('limitations',[])]
        return result
