"""Build the static showcase from frozen, path-free release inputs; no inference."""
import argparse
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'dist')
    parser.add_argument('--manifest',type=Path,default=HERE/'showcase.json')
    parser.add_argument('--media-root',type=Path,help='External folder containing assets/showcase/*.mp4 and *.jpg')
    args=parser.parse_args()
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    assets={item[key] for item in data['annotations'] for key in ('video','poster')}
    for query in data['queries']:
        assets.update(query[key] for key in ('video','poster'))
        assets.update(item[key] for item in query['candidates'] for key in ('video','poster'))
    if args.media_root:
        media_root=args.media_root.resolve()
        missing=[]
        for resource in sorted(assets):
            source=(media_root/resource).resolve()
            if not source.is_relative_to(media_root) or not source.is_file():
                missing.append(resource);continue
            destination=(output/resource).resolve()
            if not destination.is_relative_to(output):raise ValueError('Asset path escapes output')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if source!=destination:shutil.copy2(source,destination)
        if missing:raise FileNotFoundError('Missing media assets: '+', '.join(missing))
    # Without external media this is a self-contained content/layout preview.
    data['media_available']=bool(args.media_root)
    for path in (HERE/'static').rglob('*'):
        if path.is_file():
            destination=output/path.relative_to(HERE/'static');destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    for source,name in [(HERE.parent/'packages/pipeline/README.md','pipeline-readme.md'),
                        (HERE.parent/'docs/CONTINUOUS_SUBTASK_V1.md','continuous-subtask.md')]:
        shutil.copy2(source,output/'downloads'/name)
    robot=(HERE/'robotization-section.html').read_text(encoding='utf-8').strip()
    (output/'index.html').write_text((HERE/'page.html').read_text(encoding='utf-8').replace('<!-- ROBOTIZATION_UNCHANGED -->',robot),encoding='utf-8')
    (output/'showcase-data.js').write_text('window.SHOWCASE = '+json.dumps(data,ensure_ascii=False)+';\n',encoding='utf-8')
    (output/'downloads/showcase-manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'annotation_cases':len(data['annotations']),'queries':len(data['queries']),
                      'media_included':data['media_available'],'model_calls':0},ensure_ascii=False))

if __name__=='__main__':main()
