"""Read-only EgoDex reader. Explicit keys checked against actual local HDF5."""
from pathlib import Path
import json
import subprocess
import h5py
import numpy as np

HAND_KEYS = {
    side: {'wrist': f'{side}Hand', 'tips': [f'{side}ThumbTip',
        f'{side}IndexFingerTip', f'{side}MiddleFingerTip',
        f'{side}RingFingerTip', f'{side}LittleFingerTip']}
    for side in ('left', 'right')
}

def probe_video(path, fps):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=avg_frame_rate,codec_name:frame=best_effort_timestamp_time',
           '-of', 'json', str(path)]
    data = json.loads(subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8').stdout)
    stream = data['streams'][0]
    a, b = map(float, stream['avg_frame_rate'].split('/'))
    times = np.array([float(f['best_effort_timestamp_time']) for f in data['frames']])
    if len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError('missing or invalid video PTS')
    if not np.isclose(a / b, fps, atol=1e-3) or not np.allclose(np.diff(times), 1/fps, atol=1e-4):
        raise ValueError('V1 requires configured constant FPS; no implicit pose resampling')
    if abs(times[0]) > 1e-4:
        raise ValueError('nonzero initial PTS needs an explicit pose time mapping')
    return times, stream['codec_name']

def read_episode(row, cfg):
    video = Path(row['video_path'])
    if not video.is_file():
        raise FileNotFoundError(video)
    times, codec = probe_video(video, cfg['fps'])
    hands, issues = {}, {}
    with h5py.File(row['hdf5_path'], 'r') as h:
        fields = []
        h.visititems(lambda name, obj: fields.append({'key': name, 'shape': list(obj.shape),
                    'dtype': str(obj.dtype)}) if isinstance(obj, h5py.Dataset) else None)
        if 'transforms/camera' in h and len(h['transforms/camera']) != len(times):
            raise ValueError('camera/video frame count mismatch')
        attrs = {k: str(v) for k, v in h.attrs.items()}
        for side, mapping in HAND_KEYS.items():
            keys = [mapping['wrist'], *mapping['tips']]
            missing = [k for k in keys if f'transforms/{k}' not in h]
            if missing:
                issues[side] = {'missing_transforms': missing}
                continue
            poses = [np.asarray(h[f'transforms/{k}'], dtype=np.float64) for k in keys]
            if any(p.shape != (len(times), 4, 4) for p in poses):
                raise ValueError(f'{side}: pose/video shape mismatch')
            confidences, absent = {}, []
            for k in keys:
                name = f'confidences/{k}'
                if name in h:
                    c = np.asarray(h[name], dtype=np.float64)
                    if c.shape != (len(times),):
                        raise ValueError(f'confidence shape mismatch: {name}')
                    confidences[k] = c
                else:
                    absent.append(k)
            hands[side] = {'wrist': poses[0], 'tips': np.stack(poses[1:], axis=1),
                           'confidence': confidences}
            issues[side] = {'confidence_missing': absent}
    selected = attrs.get('which_llm_description', '1')
    suffix = '2' if selected in ('2', '2.0') else ''
    official = attrs.get('llm_description'+suffix) or attrs.get('description'+suffix) or row.get('task_text', '')
    return {'hands': hands, 'times': times, 'official_text': official,
            'report': {'episode_id': row['episode_id'], 'fields': fields, 'hand_mapping': HAND_KEYS,
                       'hand_issues': issues, 'frame_count': len(times), 'codec': codec,
                       'official_description_selector': selected, 'attributes': attrs}}
