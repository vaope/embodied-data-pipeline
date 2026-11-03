"""Half-open time contracts and adapters for existing event manifests."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Step:
    step_id: str
    source_id: str
    start_s: float
    end_s: float
    caption: str
    video_path: str = ''
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in (self.step_id, self.source_id, self.caption)):
            raise ValueError('step_id, source_id and caption must be nonempty strings')
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)
                and 0 <= self.start_s < self.end_s):
            raise ValueError('invalid half-open step interval')

    @classmethod
    def from_dict(cls, r):
        return cls(
            r.get('step_id', r.get('clip_id')), r.get('source_id', r.get('episode_id')),
            float(r.get('annotation_start_s', r.get('start_s', r.get('start_time')))),
            float(r.get('annotation_end_s', r.get('end_s', r.get('end_time')))),
            r['caption'], r.get('playback_video_path') or r.get('video_path', ''),
            r.get('provenance', {k: r[k] for k in (
                'annotation_source', 'annotation_path', 'annotation_sha256', 'data_warnings') if k in r}))


def validate_steps(steps):
    if not steps:
        raise ValueError('empty step sequence')
    if len({s.step_id for s in steps}) != len(steps):
        raise ValueError('duplicate step IDs')
    previous = {}
    for s in steps:
        p = previous.get(s.source_id)
        if p and s.start_s < p.end_s - 1e-7:
            raise ValueError(f'overlapping or unordered steps: {p.step_id}, {s.step_id}')
        if p and p.video_path != s.video_path:
            raise ValueError(f'inconsistent video path within source {s.source_id}')
        previous[s.source_id] = s


@dataclass(frozen=True)
class VideoReference:
    """Task-context video only: does not assert step-level time alignment."""
    source_id: str
    video_path: str
    start_s: float
    end_s: float
    note: str = '任务原视频供参考，查询步骤边界尚未核实。'

    def __post_init__(self):
        Step('reference', self.source_id, self.start_s, self.end_s, self.note, self.video_path)
        if not self.video_path:
            raise ValueError('video reference requires a video_path')


@dataclass(frozen=True)
class Query:
    query_id: str
    key_step_id: str
    steps: tuple[Step, ...]
    task: str = ''
    warnings: tuple[str, ...] = ()
    video_reference: VideoReference | None = None

    def __post_init__(self):
        validate_steps(self.steps)
        if not self.query_id or len({s.source_id for s in self.steps}) != 1:
            raise ValueError('query must have an ID and exactly one source')
        if self.key_step_id not in [s.step_id for s in self.steps]:
            raise ValueError('key_step_id must belong to query')

    @property
    def key_index(self):
        return next(i for i, s in enumerate(self.steps) if s.step_id == self.key_step_id)

    @classmethod
    def from_dict(cls, r):
        return cls(r['query_id'], r['key_step_id'], tuple(Step.from_dict(s) for s in r['steps']),
                   r.get('task', ''), tuple(r.get('warnings', [])),
                   VideoReference(**r['video_reference']) if r.get('video_reference') else None)


@dataclass(frozen=True)
class Config:
    rerank_pool_size: int = 0
    concatenated_baseline: bool = True
    anchor_top_k: int = 50
    top_k: int = 10
    neighbor_steps: int = 8
    neighbor_seconds: float = 45
    max_span_s: float = 45
    max_step_gap_s: float = 5
    max_skips: int = 2
    min_similarity: float = 0.45
    anchor_min_similarity: float = 0.45
    skip_penalty: float = 0.10
    gap_penalty_per_s: float = 0.01
    merge_penalty: float = 0.05
    nms_iou: float = 0.8
    allow_merges: bool = True

    def __post_init__(self):
        if type(self.rerank_pool_size) is not int or self.rerank_pool_size < 0:
            raise ValueError('invalid rerank pool size')
        if type(self.concatenated_baseline) is not bool:
            raise ValueError('invalid concatenated_baseline')
        for k in ('anchor_top_k', 'top_k', 'neighbor_steps', 'max_skips'):
            v = getattr(self, k)
            if type(v) is not int or v < (1 if k in ('anchor_top_k', 'top_k') else 0):
                raise ValueError(f'invalid {k}')
        for k in ('neighbor_seconds', 'max_span_s', 'max_step_gap_s', 'skip_penalty',
                  'gap_penalty_per_s', 'merge_penalty'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(f'invalid {k}')
        if self.max_span_s == 0 or not 0 < self.nms_iou <= 1:
            raise ValueError('invalid span or NMS threshold')
        for v in (self.min_similarity, self.anchor_min_similarity):
            if not math.isfinite(v) or not -1 <= v <= 1:
                raise ValueError('similarity thresholds must be in [-1,1]')
        if type(self.allow_merges) is not bool:
            raise ValueError('allow_merges must be boolean')


def load_manifest(path):
    steps = [Step.from_dict(json.loads(line)) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    steps.sort(key=lambda s: (s.source_id, s.start_s, s.end_s, s.step_id))
    validate_steps(steps)
    return steps
