"""Self-contained review page and optional continuous video excerpts."""
import json
from pathlib import Path
import subprocess


def render_clips(result, out, count):
    out = Path(out)
    records = []
    if count <= 0 or not result['complete']:
        return records
    media = out / 'media'
    media.mkdir()

    def cut(source, start, end, name):
        path = Path(source)
        if not source:
            return {'unavailable_reason': '文字查询未关联原视频。'}
        if not path.is_file():
            return {'error': 'source video unavailable', 'source': source}
        target = media / name
        command = ['ffmpeg','-nostdin','-v','error','-n','-ss',str(start),'-i',str(path),
                   '-t',str(end-start),'-an','-vf','scale=640:-2','-c:v','libx264',
                   '-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        completed = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        record = dict(command=command,source=source,start_s=start,end_s=end,exit_code=completed.returncode)
        records.append(record)
        if completed.returncode:
            record['error'] = completed.stderr[-2000:]
            return {'error':record['error']}
        return {'url':'media/'+name,'start_s':start,'end_s':end}

    q = result['query']['steps']
    reference = result['query'].get('video_reference')
    if reference:
        result['query_media'] = cut(reference['video_path'],reference['start_s'],reference['end_s'],'query.mp4')
        result['query_media'].update(context_only=True,note=reference['note'],source_id=reference['source_id'])
    else:
        result['query_media'] = cut(q[0]['video_path'],q[0]['start_s'],q[-1]['end_s'],'query.mp4')
    for rank, row in enumerate(result['complete'][:count],1):
        row['media'] = cut(row['video_path'],row['start_s'],row['end_s'],f'candidate_{rank:02d}.mp4')
    return records


def write_report(result, path):
    # JSON is embedded as data, never as markup or executable source.
    payload = json.dumps(result,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template = Path(__file__).with_name('report.html').read_text(encoding='utf-8')
    Path(path).write_text(template.replace('__RESULT_JSON__',payload),encoding='utf-8')
