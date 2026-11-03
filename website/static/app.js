const $ = id => document.getElementById(id);
const DATA = window.SHOWCASE;
const number = n => Number(n).toFixed(2);
const span = item => `${number(item.start_s)}–${number(item.end_s)} s`;
const text = (id, value) => { $(id).textContent = value; };
function node(tag, value, className) {
  const el = document.createElement(tag);
  if (value !== undefined) el.textContent = value;
  if (className) el.className = className;
  return el;
}
function loadVideo(video, item) {
  if(DATA.media_available===false){
    video.pause();video.hidden=true;video.removeAttribute('src');
    if(!video.parentElement.querySelector('.media-note'))video.after(node('div','演示媒体未随代码附带。按来源准备素材后，用 --media-root 构建即可播放。','media-note'));
    return;
  }
  if(video._pendingPlay) video.removeEventListener('loadedmetadata',video._pendingPlay);
  video._pendingPlay=null;
  video.pause(); video.src = item.video; video.poster = item.poster;
  video.removeAttribute('data-stop'); video.load();
}
function playRange(video, start, end) {
  if(DATA.media_available===false){text('play-status','当前为无媒体内容预览；区间与描述可直接查看。');return;}
  if(video._pendingPlay) video.removeEventListener('loadedmetadata',video._pendingPlay);
  const begin = () => {
    video._pendingPlay=null;
    video.currentTime = start; video.dataset.stop = end;
    video.play().catch(() => { text('play-status', '可使用视频自带控件播放。'); });
  };
  if (video.readyState >= 1) begin();
  else {video._pendingPlay=begin;video.addEventListener('loadedmetadata', begin, {once:true});}
}
document.querySelectorAll('video').forEach(video => {
  video.addEventListener('timeupdate', () => {
    if (video.dataset.stop && video.currentTime >= Number(video.dataset.stop)) {
      video.pause(); video.removeAttribute('data-stop');
    }
  });
  video.addEventListener('error', () => {
    if (!video.parentElement.querySelector('.media-error'))
      video.after(node('p', '视频暂时无法播放，请检查服务连接。', 'media-error'));
  });
  video.addEventListener('loadeddata',()=>video.parentElement.querySelector('.media-error')?.remove());
});
let annotationCase = DATA.annotations[0], variant = 'AV-A';
const variantDescriptions = {'AV-A':'Qwen3.8 + AUX · 分段与描述', 'AV-B':'共享 A 的分段边界 + Flora Captioner 描述', 'AV-C':'Flora TempSeg 原生边界 + Flora Captioner 描述'};
for (const item of DATA.annotations) {
  const option=node('option',item.title);option.value=item.id;$('annotation-case').append(option);
}
function selectVariant(next) {
  variant=next;
  document.querySelectorAll('[data-variant]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.variant===next)));
  text('variant-description',`${variant} · ${variantDescriptions[variant]} · ${annotationCase.variants[variant].length} 个区间`);
  const list=$('annotation-steps');list.replaceChildren();
  annotationCase.variants[variant].forEach((step,i)=>{
    const button=node('button',undefined,'step');button.dataset.start=step.start_s;button.dataset.end=step.end_s;
    button.append(node('span',`${String(i+1).padStart(2,'0')}　${span(step)}`),node('strong',step.caption));
    button.addEventListener('click',()=>playRange($('annotation-video'),step.start_s,step.end_s));list.append(button);
  });
}
function renderAnnotation() {
  loadVideo($('annotation-video'),annotationCase);
  text('annotation-source',`EgoDex · ${annotationCase.id.replace('egodex_','')}`);
  text('annotation-task',annotationCase.task);text('time',`0.00 / ${number(annotationCase.duration_s)} s`);
  $('annotation-download').href=annotationCase.download;
  const timelines=$('annotation-timelines');timelines.replaceChildren();
  for(const key of ['AV-A','AV-B','AV-C']) {
    const row=node('div',undefined,'timeline-row');row.append(node('span',key));
    const track=node('div',undefined,'timeline-track');const lanes=[];
    annotationCase.variants[key].forEach(step=>{
      let lane=lanes.findIndex(end=>end<=step.start_s+1e-6);
      if(lane===-1)lane=lanes.length;lanes[lane]=step.end_s;
      const block=node('button',undefined,`timeline-block variant-${key}`);
      block.style.left=`${100*step.start_s/annotationCase.duration_s}%`;
      block.style.width=`${100*(step.end_s-step.start_s)/annotationCase.duration_s}%`;block.style.top=`${lane*15}px`;
      block.title=`${key} ${span(step)} ${step.caption}`;block.setAttribute('aria-label',block.title);
      block.addEventListener('click',()=>{selectVariant(key);playRange($('annotation-video'),step.start_s,step.end_s);});track.append(block);
    });
    track.style.height=`${lanes.length*15+2}px`;row.append(track);timelines.append(row);
  }
  selectVariant(variant);
}
$('annotation-case').addEventListener('change',()=>{annotationCase=DATA.annotations.find(x=>x.id===$('annotation-case').value);renderAnnotation();});
document.querySelectorAll('[data-variant]').forEach(button=>button.addEventListener('click',()=>selectVariant(button.dataset.variant)));
$('annotation-video').addEventListener('timeupdate',()=>{
  const current=$('annotation-video').currentTime;text('time',`${number(current)} / ${number(annotationCase.duration_s)} s`);
  const active=annotationCase.variants[variant].filter(s=>current>=s.start_s&&current<s.end_s);
  text('annotation-clock',active.map(s=>s.caption).join(' / ')||'当前时刻无该路线标注');
  $('annotation-steps').querySelectorAll('.step').forEach(button=>button.classList.toggle('active',current>=Number(button.dataset.start)&&current<Number(button.dataset.end)));
});
let query,candidate,route='robot';
function renderFramework() {
  const ego=query.type==='ego';
  const legacy=query.method==='event_v1_fusion';
  const steps=ego ? [
    ['组织查询表示','任务文本 · 起始手型 · 双手运动','从查询窗口读取文本和可用手部特征。当前 Ego 样例文本为整集任务描述，手型与运动来自该窗口。'],
    ['手型＋语言联合召回','合并两路排名，取回候选窗口','分别按语言相似度和起始手型距离排序，合并排名形成候选池；这条路线并非只用语言召回。'],
    ['按活动量比较运动','位移 · 转动 · 指尖 · 腕部','比较四路运动特征，并按查询中左右手的活动量加权，突出主要运动手；候选必须具备查询所需的手部数据。'],
    ['合并排名与来源去重','控制每个来源数量，返回结果','将四路运动距离分别排序后合并名次，再限制同一来源的返回数量。下方保留原始 Top 3；这些距离不等于准确率。']
  ] : [
    ['明确查询步骤','描述目标，确定视频区间','使用步骤描述表达检索目标，并裁出对应的机器人示范区间。查询范围必须先明确：放盖单步与拿盖后再放盖是不同目标。'],
    ['按语义找候选',legacy?'相关事件池 · 旧版来源配额':'步骤语义 Top 50',legacy?'旧放笔案例按事件描述召回固定候选池，并有每来源与任务族配额；原融合结果保留在下方，不冒充新版运行。':'先保留语义相关的前 50 个步骤事件，候选池数量与最终返回数量分开；运动只在已经召回的候选内调整排序。'],
    ['末端运动小幅重排','速度 · 转动 · 停顿 · 路径比例',legacy?'对机器人末端与人腕的运动轮廓计算行为距离，在语义分上施加有界惩罚。这是旧放笔示例采用的融合版本。':'比较观测机器人末端与人腕的运动轮廓，在语义分上作有界调整；缺失轨迹保留语义分并记录原因。XYZ 方向尚未校准，不当作有效方向匹配证据。'],
    ['定位区间并复查','步骤范围 · 原排名 · 来源',legacy?'下方精选旧融合排名第 1、2 项，保留原事件区间。展示选择不改变原排序，也不代表整批检索准确率。':'从重排后的原结果中选择可复查片段。放盖例仅展示关键步骤结果，不自动声称覆盖前面的拿盖步骤。']
  ];
  text('framework-route',ego?'Ego → Ego':legacy?'Robot → Ego · 旧版事件示例':'Robot → Ego · 单步运动重排');
  text('framework-summary',ego?'先同时考虑“在做什么”和“起始手型像不像”，再比较“怎么动”。':'先找“在做什么”相关的片段，再用“怎么动”小幅调整候选顺序。');
  text('framework-scope',ego?'当前路线：pose-language-active-v1。流程图说明已有实现，不暗示学习到了跨具身统一表征。':legacy?'当前放笔示例来自旧版事件检索；切到放盖示例可查看新版语义 Top 50 后重排的路线。':'当前放盖示例是开发查询的单步结果；运动重排已有实现，尚无总体正确率提升结论。');
  const parent=$('framework-steps');parent.replaceChildren();
  const activate=index=>{
    parent.querySelectorAll('button').forEach((button,i)=>button.setAttribute('aria-pressed',String(index===i)));
    text('framework-detail-label',`${String(index+1).padStart(2,'0')} / ${steps[index][0]}`);
    text('framework-detail',steps[index][2]);
  };
  steps.forEach((step,index)=>{
    const button=node('button',undefined,'framework-step');button.setAttribute('aria-pressed','false');
    button.append(node('small',String(index+1).padStart(2,'0')),node('strong',step[0]),node('span',step[1]));
    button.addEventListener('click',()=>activate(index));parent.append(button);
  });activate(0);
}
function renderSteps(target,item,video) {
  const parent=$(target);parent.replaceChildren();
  item.steps.forEach(step=>{
    const button=node('button',`${span(step)}　${step.caption}`,'locate');
    button.addEventListener('click',()=>playRange(video,step.start_s-item.display_start_s,step.end_s-item.display_start_s));parent.append(button);
  });
}
function addField(parent,key,value){parent.append(node('dt',key),node('dd',value));}
function renderCandidate(next) {
  candidate=next;loadVideo($('result-video'),candidate);text('result-rank',`原排名 #${candidate.rank}`);
  text('result-caption',candidate.caption);text('result-span',`原片 ${span(candidate)} · ${candidate.caption_scope}`);
  renderSteps('result-steps',candidate,$('result-video'));
  $('candidates').querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.id===candidate.id)));
  $('sample-json').href=candidate.download;$('sample-csv').href=candidate.csv;
  const fields=$('sample-fields');fields.replaceChildren();
  addField(fields,'来源',candidate.source_id);addField(fields,'原片区间',span(candidate));addField(fields,'文本范围',candidate.caption_scope);
  addField(fields,'运动信息',candidate.motion?'四路特征距离可查':candidate.behavior_parts?'末端行为距离可查':'已参与原排序；详见方法记录');addField(fields,'复核状态','精选待复核 · 未标记用户验收');
  const evidence=$('retrieval-evidence');evidence.replaceChildren();
  evidence.append(node('p',`${query.recipe} · ${query.method}`),node('p',query.query_note),node('p',`源结果 SHA256：${query.source_result_sha256}`));
  if(candidate.motion){
    evidence.append(node('p','以下为原结果保存的特征距离，用于复查排序；数值不代表准确率。'));
    const table=node('table');const head=node('tr');['运动特征','特征距离','该通道排名'].forEach(s=>head.append(node('th',s)));table.append(head);
    Object.entries(candidate.motion).forEach(([key,value])=>{const row=node('tr');[key,Number(value).toPrecision(5),candidate.motion_ranks[key]].forEach(s=>row.append(node('td',String(s))));table.append(row);});evidence.append(table);
  }else if(candidate.behavior_parts){
    const table=node('table');const head=node('tr');['末端行为特征','特征距离'].forEach(s=>head.append(node('th',s)));table.append(head);
    Object.entries(candidate.behavior_parts).forEach(([key,value])=>{const row=node('tr');row.append(node('td',key),node('td',Number(value).toPrecision(5)));table.append(row);});evidence.append(table);
  }else evidence.append(node('p','本页保留原方法与排序来源；这个导出快照未包含逐项运动距离。'));
}
function renderQuery(){
  query=DATA.queries.find(x=>x.id===$('query-case').value);loadVideo($('query-video'),query);$('result-video').pause();
  renderFramework();
  text('query-title',query.title);text('query-granularity',query.granularity);text('route-description',query.method_label);text('coverage-note',query.coverage_note);
  text('query-label',`QUERY / ${query.type.toUpperCase()}`);text('query-source',query.source_id);text('query-caption',query.caption);text('query-span',`原片 ${span(query)}`);
  renderSteps('query-steps',query,$('query-video'));$('play-context').hidden=query.display_start_s===query.start_s&&query.display_end_s===query.end_s;
  text('candidate-note',`${query.candidates.length} 个精选结果 · 保留原排名`);const cards=$('candidates');cards.replaceChildren();
  query.candidates.forEach(item=>{
    const button=node('button',undefined,'candidate');button.dataset.id=item.id;button.setAttribute('aria-pressed','false');
    const img=document.createElement('img');if(DATA.media_available!==false)img.src=item.poster;else img.hidden=true;img.alt=`候选 ${item.rank} 画面`;img.loading='lazy';
    button.append(img,node('small',`原排名 #${item.rank} · ${span(item)}`),node('strong',item.caption),node('span',item.source_id));
    button.addEventListener('click',()=>renderCandidate(item));cards.append(button);
  });renderCandidate(query.candidates[0]);
}
function selectRoute(next){
  document.querySelectorAll('video').forEach(video=>video.pause());route=next;
  $('robot-tab').setAttribute('aria-pressed',String(route==='robot'));$('ego-tab').setAttribute('aria-pressed',String(route==='ego'));$('query-case').replaceChildren();
  DATA.queries.filter(x=>x.type===route).forEach(item=>{const option=node('option',item.title);option.value=item.id;$('query-case').append(option);});renderQuery();
}
$('robot-tab').addEventListener('click',()=>selectRoute('robot'));$('ego-tab').addEventListener('click',()=>selectRoute('ego'));$('query-case').addEventListener('change',renderQuery);
$('play-pair').addEventListener('click',()=>{playRange($('query-video'),query.local_start_s,query.local_end_s);playRange($('result-video'),candidate.local_start_s,candidate.local_end_s);});
$('play-context').addEventListener('click',()=>{[$('query-video'),$('result-video')].forEach(video=>{video.removeAttribute('data-stop');video.currentTime=0;video.play().catch(()=>{});});});
$('copy').addEventListener('click',async function(){try{await navigator.clipboard.writeText($('commands').textContent);this.textContent='已复制';}catch{const range=document.createRange();range.selectNodeContents($('commands'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);this.textContent='已选中，请手动复制';}});
renderAnnotation();selectRoute('robot');
