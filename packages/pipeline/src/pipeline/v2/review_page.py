"""Review page markup and keyboard-driven labelling UI (served by pipeline.v2.review)."""

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>检索复查</title><style>
body{margin:0;background:#0f141b;color:#dce6f2;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;background:#17202b;padding:10px 16px;z-index:3;border-bottom:1px solid #2a3646}
header label{margin-right:14px;color:#9fb0c6;font-size:13px}
main{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;padding:16px}
#player{position:sticky;top:74px;align-self:start}
video{width:100%;background:#000;border-radius:8px}
video.robot{width:100%;max-height:220px;background:#000;border-radius:8px;margin:6px 0}
.card{background:#17202b;border:1px solid #2a3646;border-radius:10px;padding:10px 12px;margin:8px 0;cursor:pointer}
.card.active{border-color:#4f8cff;box-shadow:0 0 0 1px #4f8cff inset}
.card .row{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.rank{color:#8fb3ff;font-weight:700}.score{color:#9fb0c6}
.task{color:#cfe0f5} .meta{color:#7f93aa;font-size:12px}
button{background:#22314a;color:#dce6f2;border:1px solid #35507a;border-radius:6px;padding:4px 10px;cursor:pointer;margin-right:6px}
button.ok{background:#1d4d33;border-color:#2f7a52}button.no{background:#4d1f28;border-color:#7a3140}
select{background:#22314a;color:#dce6f2;border:1px solid #35507a;border-radius:6px;padding:3px}
.done{opacity:.55}
.hidden{display:none}
a{color:#8fb3ff}
#help{display:none;background:#1b2635;border:1px solid #2a3646;border-radius:8px;padding:8px 12px;margin:8px 16px;font-size:13px;color:#b9c9dd}
#help b{color:#dce6f2}
</style>
<header><b>检索复查</b> <span id="stat">loading…</span>
<label><input type="checkbox" id="sliceOnly" checked> 只播切片</label>
<label><input type="checkbox" id="autoNext" checked> 标完自动下一条</label>
<label><input type="checkbox" id="onlyUnlabeled"> 只看未标</label>
<span style="color:#6f8299;font-size:12px">按 <b>?</b> 看快捷键</span> ·
<a href="/labels.csv">导出 labels.csv</a></header>
<div id="help"></div>
<main><div id="player"><video id="v" controls preload="metadata"></video>
<p class="meta" id="now">点击右侧候选加载片段</p></div><div id="list"></div></main>
<script>
const CATS=["","same_verb_wrong_object","same_object_wrong_action","motion_similar_no_contact","different_task","unclear","opposite_direction"];
let data=null, labels={}, items=[], cards=[], idx=-1;
const video=document.getElementById('v');
let sliceEnd=Infinity;
const sliceOnly=document.getElementById('sliceOnly');
const autoNext=document.getElementById('autoNext');
const onlyUnlabeled=document.getElementById('onlyUnlabeled');
const HELP='快捷键：<b>Y</b> 相关 · <b>N</b> 不相关 · <b>1–6</b> 错误类别 · <b>↑/↓</b> 上一条/下一条 · <b>空格</b> 播放/暂停 · <b>R</b> 重播 · <b>L</b> 切换只播切片 · <b>A</b> 切换自动下一条 · <b>U</b> 跳到未标 · <b>?</b> 关闭本帮助';
async function boot(){
  data=await (await fetch('/candidates.json')).json();
  labels=await (await fetch('/labels.json')).json();
  const list=document.getElementById('list');
  for(const q of data.queries){
    const h=document.createElement('h3'); h.textContent=q.query_id+" · "+q.instruction; list.appendChild(h);
    const meta=document.createElement('div'); meta.className='meta';
    meta.textContent=`${q.eef.steps} steps / ${q.eef.duration_s}s / path ${q.eef.path_length_m.toFixed(3)}m`;
    list.appendChild(meta);
    if(q.robot_video){
      const rv=document.createElement('video'); rv.className='robot'; rv.controls=true; rv.loop=true; rv.muted=true;
      rv.preload='metadata'; rv.src='/robot/'+encodeURIComponent(q.query_id);
      rv.title='DROID robot demonstration (three cameras)';
      list.appendChild(rv);
    }
    for(const c of q.candidates){
      const id=q.query_id+"|"+c.clip_id, lab=labels[id];
      const card=document.createElement('div'); card.className='card'+(lab!==undefined?' done':'');
      card.innerHTML=`<div class="row"><span class="rank">#${c.rank}</span><span class="score">sim ${c.similarity.toFixed(4)}</span>
        <span class="task">${(c.task_text||'').slice(0,110)}</span></div>
        <div class="meta">${c.episode_id} · ${c.start_time.toFixed(1)}–${c.end_time.toFixed(1)}s · ${c.clip_id}</div>
        <div style="margin-top:6px"><button class="ok">相关</button><button class="no">不相关</button>
        <select>${CATS.map(x=>`<option value="${x}">${x||'错误类别(可选)'}</option>`).join('')}</select></div>`;
      const item={id, qid:q.query_id, c};
      card.querySelector('.ok').onclick=()=>mark(item,1,card);
      card.querySelector('.no').onclick=()=>mark(item,0,card);
      card.onclick=(ev)=>{ if(ev.target.tagName==='BUTTON'||ev.target.tagName==='SELECT'||ev.target.tagName==='OPTION') return;
        selectCard(cards.indexOf(card)); };
      list.appendChild(card); items.push(item); cards.push(card);
    }
  }
  video.addEventListener('timeupdate',()=>{
    if(sliceOnly.checked && video.currentTime>=sliceEnd){ video.pause(); }
  });
  updateStat();
  applyFilter();
  onlyUnlabeled.onchange=applyFilter;
  if(cards.length) selectCard(0);
  window.addEventListener('keydown',onKey);
}
function loadItem(item,autoplay){
  video.onerror=()=>{ document.getElementById('now').textContent='视频加载失败 error='+(video.error&&video.error.code)+' '+item.c.clip_id; };
  video.src='/media/'+encodeURIComponent(item.c.clip_id);
  sliceEnd=item.c.end_time;
  video.onloadedmetadata=()=>{ video.currentTime=Math.max(0,item.c.start_time); if(autoplay) video.play().catch(()=>{}); };
  document.getElementById('now').textContent=`${item.c.clip_id} · ${item.c.start_time.toFixed(2)}–${item.c.end_time.toFixed(2)}s`;
}
function selectCard(i){
  if(i<0||i>=cards.length) return;
  idx=i; cards.forEach((c,k)=>c.classList.toggle('active',k===i));
  cards[i].scrollIntoView({block:'nearest'});
  loadItem(items[i],true);
}
async function mark(item,label,card){
  const cat=card.querySelector('select').value;
  await fetch('/label',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({id:item.id,label,category:cat})});
  labels[item.id]=label; card.classList.add('done'); updateStat(); applyFilter();
  if(autoNext.checked) selectCard(Math.min(idx+1,cards.length-1));
}
function applyFilter(){
  cards.forEach(c=>c.classList.toggle('hidden', onlyUnlabeled.checked && c.classList.contains('done')));
}
function nextUnlabeled(){
  for(let k=idx+1;k<cards.length;k++){ if(!cards[k].classList.contains('done')){ selectCard(k); return; } }
  for(let k=0;k<cards.length;k++){ if(!cards[k].classList.contains('done')){ selectCard(k); return; } }
}
function updateStat(){
  document.getElementById('stat').textContent=`已标 ${document.querySelectorAll('.card.done').length} / ${cards.length}`;
}
function setCategory(k){
  const card=cards[idx]; if(!card||k<1||k>=CATS.length) return;
  const sel=card.querySelector('select'); sel.value=CATS[k];
}
function onKey(e){
  if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT') return;
  const k=e.key;
  if(k==='y'||k==='Y'){ if(idx>=0) mark(items[idx],1,cards[idx]); e.preventDefault(); }
  else if(k==='n'||k==='N'){ if(idx>=0) mark(items[idx],0,cards[idx]); e.preventDefault(); }
  else if(k>='1'&&k<='6'){ setCategory(parseInt(k,10)); e.preventDefault(); }
  else if(k==='ArrowDown'||k==='j'){ selectCard(Math.min(idx+1,cards.length-1)); e.preventDefault(); }
  else if(k==='ArrowUp'||k==='k'){ selectCard(Math.max(idx-1,0)); e.preventDefault(); }
  else if(k===' '){ if(video.paused) video.play(); else video.pause(); e.preventDefault(); }
  else if(k==='r'||k==='R'){ if(idx>=0){ video.currentTime=items[idx].c.start_time; video.play().catch(()=>{}); } e.preventDefault(); }
  else if(k==='l'||k==='L'){ sliceOnly.checked=!sliceOnly.checked; e.preventDefault(); }
  else if(k==='u'||k==='U'){ nextUnlabeled(); e.preventDefault(); }
  else if(k==='a'||k==='A'){ autoNext.checked=!autoNext.checked; e.preventDefault(); }
  else if(k==='?'){ const h=document.getElementById('help'); const show=h.style.display!=='block';
    h.style.display=show?'block':'none'; h.innerHTML=HELP; e.preventDefault(); }
}
boot();
</script></html>"""
