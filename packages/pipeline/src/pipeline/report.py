"""Human-readable evidence from real reader results; static local HTML and PNGs."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import read_json, write_json


def _font(size=20):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)


def _save_image(rgb: np.ndarray,path: Path,label: str):
    image=Image.fromarray(rgb)
    image.thumbnail((768,432))
    canvas=Image.new('RGB',(768,484),'#101b2b');canvas.paste(image,(0,52))
    ImageDraw.Draw(canvas).text((16,14),label,font=_font(19),fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def _style():
    return '''<style>:root{color-scheme:dark}body{margin:0;background:#0b1220;color:#dbe4f2;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:40px auto;padding:0 24px}h1,h2,h3{color:#fff;font-weight:650;overflow-wrap:anywhere}h1{font-size:32px}a{color:#7dd3fc}.note{border-left:4px solid #fbbf24;padding:16px;background:#152238;border-radius:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:20px}.card{min-width:0;background:#152238;border:1px solid #293c57;padding:22px;border-radius:10px}.card h2{font-size:20px}img{max-width:100%;border-radius:6px}table{border-collapse:collapse;width:100%;margin:20px 0;table-layout:fixed}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #293c57;padding:10px;overflow-wrap:anywhere}th{color:#9cbbd9}code,pre{background:#0a1526;color:#c7eaff}pre{padding:16px;overflow:auto;border-radius:6px}.muted{color:#99acc4}li{margin:6px 0}.badge{display:inline-block;padding:3px 10px;background:#394026;border:1px solid #8c7b32;border-radius:20px;font-size:14px}details{margin:16px 0}summary{cursor:pointer;color:#7dd3fc}</style>'''


def _json_html(value):
    return html.escape(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def episode_report(directory: Path,bundle: dict,evidence: dict) -> None:
    kind={'source_rgb':'原视频','depth_composite_rgb':'机器人深度合成'}.get(evidence['observation_kind'],evidence['observation_kind'])
    _save_image(evidence['image_rgb'],directory/'previews/reader-source.png',f"{kind} · 帧 {evidence['source_frame_index']} · {evidence['timestamp_s']:.6f}s")
    _save_image(evidence['next_image_rgb'],directory/'previews/reader-next.png',f"{kind} · 下一帧 {evidence['target_frame_index']} · {evidence['next_timestamp_s']:.6f}s")
    quality=read_json(Path(bundle['quality']['report_ref']['path']))
    index=read_json(Path(bundle['visual']['index_ref']['path']))
    extras=[]
    for kind,label in [('robot_only_rgb','已有机器人渲染'),('arm_removed_background','已有去臂预览（非最终训练图像）'),('depth_composite_rgb','深度合成候选')]:
        frames=[r for r in index['frames'] if r['image_kind']==kind]
        if not frames:continue
        row=frames[len(frames)//2]
        if kind == 'arm_removed_background':
            label='去臂预览（待补正式结果）' if row.get('preview_only',True) else '真实去臂结果（验收状态见清单）'
        if kind == 'depth_composite_rgb':
            label='深度合成候选（待验收）' if row.get('preview_only',True) else '已审阅的深度合成'
        rgb=np.asarray(Image.open(row['path']).convert('RGB'))
        name=f'previews/{kind}.png'
        _save_image(rgb,directory/name,f'{label} · 帧 {row["frame_index"]}')
        extras.append((label,name))
    if bundle['refs']['canonical'].get('path'):
        with np.load(bundle['refs']['canonical']['path'],allow_pickle=False) as data:
            requested=data['gripper_width_m'][:,0]
        with np.load(bundle['refs']['robot']['path'],allow_pickle=False) as data:
            bound=data['gripper_qpos'].sum(axis=1)
        _gripper_figure(requested,bound,directory/'previews/gripper-width.png')
        extras.append(('目标宽度与机器人实际绑定宽度','previews/gripper-width.png'))
    action=evidence['action7']
    action_json=action.tolist() if np.isfinite(action).all() else None
    step=evidence['step_meta']
    title=bundle['episode_id']
    reasons=''.join(f'<li>{html.escape(reason)}</li>' for reason in bundle['readiness_reasons'])
    stage_rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(value["program_status"])}</td><td>{html.escape(json.dumps(value["counts"],ensure_ascii=False))}</td><td>{html.escape(", ".join(value["reason_codes"]))}</td></tr>' for name,value in bundle['module_status'].items())
    cards=''.join(f'<div class="card"><h3>{html.escape(label)}</h3><img src="{name}" alt="{html.escape(label)}"></div>' for label,name in extras)
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · Stage 5</title>{_style()}<main><p><a href="../index.html">← 全部样本</a></p><h1>{title}</h1><p><span class="badge">{bundle['training_readiness']}</span>　{bundle['counts']['frames']} 帧 / {bundle['counts']['transitions']} 个相邻转换</p><div class="note">当前可检查数据与接口。正式训练样本：{bundle['counts']['training_transitions']}。原视频、机器人渲染与去臂预览各自标明来源；未运行新模型，未把预览当作视觉验收通过。</div><h2>Reader 实际返回</h2><div class="grid"><img src="previews/reader-source.png" alt="源帧"><img src="previews/reader-next.png" alt="目标帧"></div><p>动作 profile：<code>{html.escape(evidence['action_profile'])}</code>；前3维 m/转换、中3维 rad/转换、第7维为下一帧可实现开度 m。</p><pre>{_json_html({'source_frame':evidence['source_frame_index'],'target_frame':evidence['target_frame_index'],'action7':action_json,'transition_valid':evidence['transition_valid'],'reason_codes':evidence['reason_codes'],'step_meta':step})}</pre><h2>当前未满足的条件</h2><ul>{reasons}</ul><h2>逐模块结果</h2><table><tr><th>模块</th><th>程序状态</th><th>计数</th><th>原因</th></tr>{stage_rows}</table><h2>已有产物与动作诊断</h2><div class="grid">{cards}</div><details><summary>逐帧 QC 与规则覆盖</summary><pre>{_json_html(quality)}</pre></details><p><a href="bundle.json">统一数据包清单</a> · <a href="quality/report.json">QC 明细</a> · <a href="export/episode.manifest.json">HDF5 导出清单</a> · <a href="export/frames.jsonl">逐帧元数据</a></p><p class="muted">Stage 3 Schema v1；Dedup / Rebalance 保持 hold；检查结果不代表用户验收。</p></main></html>'''
    content=content.replace('原视频、机器人渲染与去臂预览各自标明来源；未运行新模型，未把预览当作视觉验收通过。',
                            '图像种类、真实模型来源和执行记录保存在清单中；模型计算与用户视觉验收分别记录。')
    content=content.replace('Stage 3 Schema v1；Dedup / Rebalance 保持 hold；检查结果不代表用户验收。',
                            'Stage 3 Schema v1；监督范围：'+html.escape(bundle.get('supervision_profile','stage3_steps_strict.v1'))+'；Dedup / Rebalance 保持 hold；检查结果不代表用户验收。')
    for name,label in [('episode.manifest.json','HDF5 导出清单'),('frames.jsonl','逐帧元数据')]:
        if not (directory/'export'/name).is_file():
            content=content.replace(f' · <a href="export/{name}">{label}</a>','')
    (directory/'report.html').write_text(content,encoding='utf-8')
    (directory/'report.md').write_text(f'# {title}\n\n状态：{bundle["training_readiness"]}。{bundle["counts"]["frames"]} 帧，{bundle["counts"]["transitions"]} transitions，正式训练 {bundle["counts"]["training_transitions"]}。\n\nReader代表帧：![source](previews/reader-source.png)\n\n未满足项：\n'+''.join('- '+r+'\n' for r in bundle['readiness_reasons']),encoding='utf-8')


def _gripper_figure(requested,bound,path):
    image=Image.new('RGB',(1100,420),'#101b2b');d=ImageDraw.Draw(image)
    d.text((35,20),'夹爪宽度：青色=canonical 目标；黄色=Panda 实际绑定；虚线=0.08 m上限',font=_font(20),fill='white')
    lo,hi=0.,max(.12,float(np.max(requested))*1.05)
    xy=lambda i,v:(60+i*990/max(len(requested)-1,1),360-(v-lo)/(hi-lo)*265)
    for value in [.0,.04,.08,.12]:
        y=xy(0,value)[1];d.line((60,y,1050,y),fill='#334255');d.text((4,y-10),f'{value:.2f}',font=_font(16),fill='#9fb8d4')
    for a,color in [(requested,'#67e8f9'),(bound,'#fcd34d')]:d.line([xy(i,float(v)) for i,v in enumerate(a)],fill=color,width=3)
    d.text((65,380),'源视频 frame_index（不删帧、不重采样）',font=_font(17),fill='#9fb8d4')
    image.save(path)


def run_report(output: Path,summary: dict):
    cards=[]
    for row in summary['episodes']:
        title=html.escape(row['episode_id'])
        if row['status']=='failed':
            cards.append(f'<div class="card"><h2>{title}</h2><span class="badge">failed</span><pre>{html.escape(row["error"])}</pre></div>');continue
        counts=row['counts']
        cards.append(f'<div class="card"><h2>{title}</h2><p><span class="badge">{row["training_readiness"]}</span>　{row["status"]}</p><p>{counts["frames"]}帧 · {counts["transitions"]}个相邻转换 · {counts["finite_action_transitions"]}个可读动作 · {counts["training_transitions"]}个正式训练转换</p><a href="{title}/report.html"><img src="{title}/previews/reader-source.png" alt="reader真实读取的原视频帧">查看该样本的接口、图像、动作与QC →</a></div>')
    n=sum(r.get('counts',{}).get('frames',0) for r in summary['episodes'])
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>学习 Stage 5 · Pipeline 运行报告</title>{_style()}<main><p class="muted">学习 Stage 5 · 2026-10-19 · {html.escape(summary['run_id'])}</p><h1>Pipeline 接口已经连通到数据包与 Reader</h1><div class="note">本轮 {len(summary['episodes'])} 条 / {n} 帧。程序完成与正式训练准入分开记录：真实场景深度、完整视觉验收尚有依赖，所以当前输出为候选数据包。Stage 1–3 原结果与复现目录保持只读；无新增模型调用或IK。</div><h2>固定小样本</h2><div class="grid">{''.join(cards)}</div><h2>可复查内容</h2><ul><li>源视频/HDF5哈希、真实PTS、既有几何产物与完整图像索引。</li><li>Schema v1原标注保真、逐帧Step关联、未验证物体ID与Unknown传播。</li><li>逐帧QC/reason、N−1动作、源帧/目标帧和夹爪实际绑定开度。</li><li>自定义HDF5含RGB、hand/canonical/robot、QC、动作与语言；JSONL末帧无outgoing action。</li></ul><p><a href="run-report.json">运行清单与计数</a></p><p class="muted">本页中“可读动作”表示数组有限且可检查，不表示该转换已满足训练质量条件。Stage4 Dedup/Rebalance仍hold；未使用Mock代替真实模型。</p></main></html>'''
    date=str(summary.get('created_at',''))[:10] or '运行日期见清单'
    content=content.replace('学习 Stage 5 · 2026-10-19 ·','学习 Stage 5 · '+html.escape(date)+' ·')
    content=content.replace('真实场景深度、完整视觉验收尚有依赖，所以当前输出为候选数据包。',
                            '各样本的实际依赖与验收状态见下方清单，只有所有必需条件满足后才提供正式训练转换。')
    content=content.replace('无新增模型调用或IK。',
                            '本轮入口模式：'+html.escape(summary.get('mode','reuse-only'))+'；模型执行与外部产物来源详见各样本manifest。')
    (output/'index.html').write_text(content,encoding='utf-8')
