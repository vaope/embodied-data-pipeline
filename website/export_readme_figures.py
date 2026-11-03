"""Render research figures from the same frozen data/media used by the HTML.

No inference, generated imagery, fabricated ranking or model score is involved.
Requires matplotlib, Pillow and FFmpeg; media stays an explicit local input.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import io
import json
import subprocess
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image
from matplotlib import font_manager
from visuals_i18n import translate

HERE=Path(__file__).resolve().parent
LANG='en'
PAPER='#f6f7f2';INK='#193d35';MUTED='#61756b';GREEN='#4f8970';GOLD='#b38b53';BLUE='#688eaa'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':MUTED})

def tr(text):return translate(text,LANG)

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def frame(root,item,time,crop=False):
    source=root/item['video']
    result=subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(time),'-i',str(source),
                           '-frames:v','1','-f','image2pipe','-c:v','png','pipe:1'],check=True,capture_output=True)
    image=Image.open(io.BytesIO(result.stdout)).convert('RGB')
    if crop:
        w,h=image.size;image=image.crop((w//3,0,2*w//3,h))
    return image

def fig(height):
    f=plt.figure(figsize=(14.8,height),facecolor=PAPER)
    return f

def title(f,kicker,heading,sub):
    f.text(.035,.955,tr(kicker),color=GREEN,size=10,weight='bold')
    f.text(.035,.902,tr(heading),size=24,weight='bold')
    f.text(.035,.855,tr(sub),color=MUTED,size=10.5)

def picture(f,image,box):
    ax=f.add_axes(box);ax.imshow(image);ax.axis('off');return ax

def save(f,path):
    if LANG=='zh-CN':path=path.with_name(path.stem+'.zh-CN'+path.suffix)
    f.savefig(path,dpi=160,facecolor=PAPER);plt.close(f)

def annotation_figure(data,media,out):
    case=data['annotations'][0]
    f=fig(7.9)
    title(f,'01 / VIDEO UNDERSTANDING','One video. Time-localized steps.',
          'Real EgoDex episode / same AV-A, AV-B and AV-C outputs as the interactive review page')
    selected=case['variants']['AV-A'][:4]
    for i,step in enumerate(selected):
        t=(step['start_s']+step['end_s'])/2
        x=.035+i*.24
        picture(f,frame(media,case,t),[x,.493,.222,.298])
        f.text(x,.463,f"{t:05.2f} s  |  {'步骤' if LANG=='zh-CN' else 'STEP'} {i+1:02d}",size=9,color=GREEN,weight='bold')
        f.text(x,.429,textwrap.fill(tr(step['caption']),34),size=10,va='top',linespacing=1.4)
    ax=f.add_axes([.09,.145,.875,.19]);ax.set_facecolor(PAPER)
    colors={'AV-A':GREEN,'AV-B':GOLD,'AV-C':BLUE}
    for k,(variant,segments) in enumerate(case['variants'].items()):
        y=2-k;lanes=[]
        for seg in segments:
            lane=next((i for i,end in enumerate(lanes) if end<=seg['start_s']+1e-6),len(lanes))
            if lane==len(lanes):lanes.append(seg['end_s'])
            else:lanes[lane]=seg['end_s']
            ax.add_patch(Rectangle((seg['start_s'],y-.16+lane*.13),seg['end_s']-seg['start_s'],.13,
                                   facecolor=colors[variant],edgecolor=PAPER,linewidth=.6))
        ax.text(case['duration_s']+.5,y,f"{len(segments)} {'个区间' if LANG=='zh-CN' else 'intervals'}",size=8,va='center',color=MUTED)
    ax.set_xlim(0,case['duration_s']+8);ax.set_ylim(-.4,2.7)
    ax.set_yticks([2,1,0],['AV-A','AV-B','AV-C']);ax.set_xlabel(tr('Original video time (seconds)'),fontsize=9)
    ax.tick_params(axis='both',length=0,labelsize=9)
    for spine in ax.spines.values():spine.set_visible(False)
    for sec in [0,10,20,30,40,50,60]:ax.axvline(sec,color='#dbe2d7',lw=.5,zorder=0)
    f.text(.035,.047,tr('Native captions and boundaries retained, including overlaps. Automatic outputs; no semantic-accuracy claim.'),size=9,color=MUTED)
    save(f,out/'annotation-timeline.png')

def robot_figure(data,media,out):
    query=next(q for q in data['queries'] if q['id']=='robot-pen')
    f=fig(6.1)
    title(f,'02 / ROBOT TO EGO','Find human demonstrations for a robot step.',
          'Query: Place the pen on the table / selected original fusion ranks 1 and 2')
    cards=[query,*query['candidates']]
    for i,item in enumerate(cards):
        x=.035+i*.322
        local=item['local_start_s']+(item['local_end_s']-item['local_start_s'])*.55
        picture(f,frame(media,item,local,crop=i==0),[x,.32,.292,.405])
        label='ROBOT QUERY / DROID' if i==0 else f"EGO CANDIDATE / ORIGINAL #{item['rank']}"
        if LANG=='zh-CN':label=tr(label) if i==0 else f"人类候选 / 原始第 {item['rank']} 名"
        f.text(x,.767,label,size=10,color=GREEN,weight='bold')
        f.text(x,.275,textwrap.fill(tr(item['caption']),37),size=13,weight='bold',va='top')
        f.text(x,.162,f"{'原片时间' if LANG=='zh-CN' else 'Original time'}: {item['start_s']:.2f}-{item['end_s']:.2f} s",size=10,color=MUTED)
        f.text(x,.119,(tr('DROID 006 / middle camera crop') if i==0 else 'EgoDex / '+item['source_id'].split('_')[-1]),size=9,color=MUTED)
    f.text(.035,.042,tr('Related placing actions, preserved source intervals. Frame panels do not imply temporal alignment or task-level accuracy.'),size=9,color=MUTED)
    save(f,out/'robot-retrieval.png')

def ego_figure(data,media,out):
    query=next(q for q in data['queries'] if q['type']=='ego')
    f=fig(5.7)
    title(f,'03 / EGO TO EGO','Compare a human query with the retrieved windows.',
          'Stapling task / pose-language-active-v1 / the original Top 3 order is unchanged')
    cards=[query,*query['candidates']]
    for i,item in enumerate(cards):
        x=.035+i*.24
        local=(item['local_start_s']+item['local_end_s'])/2
        picture(f,frame(media,item,local),[x,.376,.222,.335])
        label='EGO QUERY' if i==0 else f"ORIGINAL RANK #{item['rank']}"
        if LANG=='zh-CN':label=tr(label) if i==0 else f"原始排名第 {item['rank']} 名"
        f.text(x,.755,label,size=10,color=GREEN,weight='bold')
        f.text(x,.323,f"{item['start_s']:.2f}-{item['end_s']:.2f} s",size=12,weight='bold')
        source=item['source_id'].replace('egodex_','').replace('part4_','')
        f.text(x,.271,textwrap.fill(source,30),size=9,color=MUTED,va='top')
        ax=f.add_axes([x,.145,.222,.046]);ax.set_xlim(0,5);ax.set_ylim(0,1);ax.axis('off')
        ax.add_patch(Rectangle((0,.2),5,.6,color='#d9e2d4'))
        ax.add_patch(Rectangle((item['local_start_s'],.2),item['local_end_s']-item['local_start_s'],.6,color=GREEN))
    f.text(.035,.069,tr('Green: 1 s matching window. Pale: playback context. Text is task-level; complete stapling steps are not inferred from a window.'),size=9,color=MUTED)
    save(f,out/'ego-retrieval.png')

def logic_figure(out):
    f=fig(4.4);ax=f.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.035,.92,tr('TWO QUERY SOURCES / TWO RETRIEVAL ROUTES'),size=11,color=GREEN,weight='bold')
    routes=[(.63,'ROBOT → EGO',[
        ('Step query','Caption + event interval'),('Semantic recall','Related candidate events'),
        ('Effector re-ranking','Speed / turn / stop / path'),('Inspect & select','Intervals + provenance')]),
        (.24,'EGO → EGO',[
        ('Human window','Language + pose + motion'),('Joint recall','Language + initial hand pose'),
        ('Motion comparison','Activity-weighted 4 routes'),('Fuse & diversify','Ranks + source limits')])]
    for y,label,steps in routes:
        ax.text(.035,y+.19,tr(label),size=11,color=GREEN,weight='bold')
        for i,(name,detail) in enumerate(steps):
            x=.035+i*.242
            ax.add_patch(FancyBboxPatch((x,y-.025),.205,.17,boxstyle='round,pad=0.009,rounding_size=.015',
                                       facecolor='#e7eee1',edgecolor='#cedbc8'))
            ax.text(x+.013,y+.085,tr(name),size=12,weight='bold')
            ax.text(x+.013,y+.02,tr(detail),size=8.7,color=MUTED)
            if i<3:ax.add_patch(FancyArrowPatch((x+.21,y+.06),(x+.237,y+.06),arrowstyle='->',mutation_scale=13,color=GREEN))
    ax.text(.035,.058,tr('Continuous steps: anchor recall → same-source ordered matching → missing-step checks → optional motion re-ranking.'),size=9,color=MUTED)
    save(f,out/'retrieval-routes.png')

def animation(data,media,out):
    query=next(q for q in data['queries'] if q['id']=='robot-pen');candidate=query['candidates'][0]
    filters=('[0:v]fps=8,crop=iw/3:ih:iw/3:0,scale=480:270,setpts=PTS-STARTPTS[a];'
             '[1:v]fps=8,scale=480:270,tpad=stop_mode=clone:stop_duration=8,setpts=PTS-STARTPTS[b];'
             '[a][b]hstack=inputs=2:shortest=1,split[x][y];'
             '[x]palettegen=stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4')
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(media/query['video']),'-i',str(media/candidate['video']),
                    '-filter_complex',filters,'-loop','0',str(out/'robot-to-ego.gif')],check=True)

def gallery(out,language='en'):
    items=[('robot-to-ego.gif','Robot and human: related placing actions','Left: DROID middle camera. Right: EgoDex candidate. Original speed; the shorter human clip holds its final frame. No time alignment.'),
           ('annotation-timeline.png','Read the task as localized steps','Frames and native A/B/C boundaries from the same frozen annotation example.'),
           ('robot-retrieval.png','Robot query and human candidates','Preserved source spans and original selected fusion ranks.'),
           ('ego-retrieval.png','Human query and original Top 3','Fixed one-second windows; source task language is not a window-level action label.'),
           ('retrieval-routes.png','How retrieval works','A compact explanation of the existing two routes, without an improvement claim.')]
    articles=[]
    for name,heading,caption in items:
        if language=='zh-CN' and name.endswith('.png'):name=name.replace('.png','.zh-CN.png')
        heading=translate(heading,language);caption=translate(caption,language)
        mime='image/gif' if name.endswith('.gif') else 'image/png'
        uri='data:'+mime+';base64,'+base64.b64encode((out/name).read_bytes()).decode()
        articles.append(f'<section><h2>{html.escape(heading)}</h2><img src="{uri}" alt="{html.escape(heading)}"><p>{html.escape(caption)}</p></section>')
    introduction=translate('Real examples exported from the interactive annotation and retrieval review. This offline visual gallery embeds all figures; no installation or server is needed.',language)
    footer=translate('Selected results for private review, not a benchmark or blanket user acceptance. Source: EgoDex / DROID and the retained annotation/retrieval runs. See visual sources in the repository. The full video review interface remains under website/.',language)
    document=f'''<!doctype html><html lang="{language}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Embodied Data Pipeline · {translate('Real examples',language)}</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#193d35;background:#f6f7f2;margin:0 auto;padding:40px 22px;max-width:1180px}}h1{{font-size:40px}}h2{{font-size:26px}}section{{padding:30px 0;border-top:1px solid #d4dfcc}}img{{width:100%;border-radius:6px}}p{{color:#61756b}}a{{color:inherit}}small{{font-size:12px}}</style></head><body><nav aria-label="Language"><a href="showcase.html" lang="en">English</a> · <a href="showcase.zh-CN.html" lang="zh-CN">简体中文</a></nav><h1>Embodied Data Pipeline</h1><p>{html.escape(introduction)}</p>'''+''.join(articles)+f'<small>{html.escape(footer)}</small></body></html>'
    name='showcase.zh-CN.html' if language=='zh-CN' else 'showcase.html'
    (out.parent/name).write_text(document,encoding='utf-8')

def main():
    global LANG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE.parent/'docs/visuals')
    parser.add_argument('--language',choices=['en','zh-CN'],default='en')
    parser.add_argument('--font',type=Path,help='CJK font file for Chinese figures; defaults to Microsoft YaHei on Windows')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    LANG=args.language
    if LANG=='zh-CN':
        font=args.font or Path('C:/Windows/Fonts/msyh.ttc')
        if not font.is_file():parser.error('Chinese figures require --font pointing to a CJK font file')
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    data=json.loads((HERE/'showcase.json').read_text(encoding='utf-8-sig'));media=args.media_root.resolve()
    annotation_figure(data,media,out);robot_figure(data,media,out);ego_figure(data,media,out);logic_figure(out)
    if LANG=='en' or not (out/'robot-to-ego.gif').is_file():animation(data,media,out)
    gallery(out,LANG)
    if LANG=='zh-CN' and all((out/name).is_file() for name in ['annotation-timeline.png','robot-retrieval.png','ego-retrieval.png','retrieval-routes.png']):gallery(out,'en')
    sources=[]
    for item in [data['annotations'][0],*(q for q in data['queries'] if q['id'] in {'robot-pen','ego-staple'})]:
        for source in [item,*item.get('candidates',[])]:
            sources.append({'id':source.get('id'),'video':source['video'],'video_sha256':digest(media/source['video']),
                            'source_span_s':[source.get('start_s'),source.get('end_s')],
                            'source_result_sha256':item.get('source_result_sha256'),'annotation_provenance':item.get('provenance')})
    record={'created_at':'2026-11-03','source_manifest_sha256':digest(HERE/'showcase.json'),
            'kind':'real-source-frame-and-data-figures; not browser screenshots',
            'languages':['en','zh-CN'] if (out/'annotation-timeline.zh-CN.png').is_file() else ['en'],
            'translation_scope':'display text only; source captions, time intervals and rankings unchanged',
            'model_calls':0,'rankings_changed':False,'public_media_clearance':'pending; current repository private',
            'sources':sources,'outputs':[{'path':p.name,'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.suffix in {'.png','.gif'}]}
    (out/'SOURCES.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures':len(record['outputs']),'total_bytes':sum(x['bytes'] for x in record['outputs']),
                      'offline_html_bytes':(out.parent/'showcase.html').stat().st_size,'out':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
