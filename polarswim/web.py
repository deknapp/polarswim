"""Local web UI: browse swims, see the analysis, copy the Strava card.

Flask serving one self-contained page — no CDN, no build step, no external
requests — plus a small JSON API the page calls. It binds to localhost only; this
is a personal tool over a personal database, not a service.

The heavy lifting all lives in `analyze` / `report` / `render`, so the UI is a thin
presentation layer over the same derivations the CLI uses.
"""

from __future__ import annotations

import datetime as dt

from flask import Flask, Response, jsonify, request

from . import ai, analyze, db, image, metrics, render, report

# Web equivalents of the card's emoji palette, so the dashboard and the pasted
# card describe a stroke with the same colour.
PIE_COLORS = {
    "freestyle": "#4aa3ff", "backstroke": "#3ddc84", "breaststroke": "#f0883e",
    "butterfly": "#bc7cff", "other": "#c9d1d9", "undetermined": "#6b7280",
}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>polarswim</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0f1419;--panel:#171d24;--line:#252d36;--fg:#e6edf3;--dim:#8b949e;
       --accent:#4aa3ff;--warn:#f0a848}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:var(--bg);color:var(--fg)}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
        gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:16px;margin:0;letter-spacing:.5px}
 .dim{color:var(--dim)}
 .wrap{display:flex;min-height:calc(100vh - 53px);flex-wrap:wrap}
 aside{width:260px;border-right:1px solid var(--line);overflow-y:auto;
       max-height:calc(100vh - 53px)}
 main{flex:1;padding:20px;min-width:320px}
 .sw{padding:9px 14px;border-bottom:1px solid var(--line);cursor:pointer}
 .sw:hover{background:var(--panel)} .sw.on{background:var(--panel);
       border-left:3px solid var(--accent)}
 .sw b{display:block;font-weight:600} .sw span{color:var(--dim);font-size:12px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
       padding:16px;margin-bottom:16px}
 pre{font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;
     overflow-x:auto;margin:0}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--dim);font-weight:500}
 button{background:var(--accent);color:#04121f;border:0;border-radius:6px;
        padding:7px 13px;font-weight:600;cursor:pointer;margin-right:8px}
 button.alt{background:var(--line);color:var(--fg)}
 .lo{color:var(--warn)}
 .chip{display:inline-block;min-width:24px;text-align:center;border-radius:4px;
       padding:1px 5px;color:#08131f;font-weight:700;font-size:11px}
 .pctbar{display:inline-block;width:46px;height:7px;background:var(--line);
         border-radius:3px;overflow:hidden;vertical-align:middle;margin-right:5px}
 .pctbar i{display:block;height:100%}
 .pr{color:#f0a848;font-weight:700}
 /* A bests table has few, short columns; letting it fill a wide window strands
    the date a screen away from the time it belongs to. */
 .narrow{max-width:720px}
 .tab{background:transparent;color:var(--dim);border:1px solid transparent;
      font-weight:600;padding:6px 12px;margin-right:4px}
 .tab.on{background:var(--panel);color:var(--fg);border-color:var(--line)}
 .stat{display:inline-block;min-width:150px;margin:0 26px 18px 0;vertical-align:top}
 .stat b{display:block;font-size:26px;font-weight:700;line-height:1.25}
 .stat span{color:var(--dim);font-size:12px;text-transform:uppercase;
            letter-spacing:.6px}
 .zrow{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px}
 .ztrack{flex:1;height:12px;background:var(--line);border-radius:3px;overflow:hidden}
 .ztrack i{display:block;height:100%}
 h2{font-size:14px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;
    margin:0 0 12px}
 .bar{height:9px;background:var(--accent);border-radius:2px;display:inline-block}
 input[type=date]{background:var(--panel);color:var(--fg);border:1px solid var(--line);
        border-radius:5px;padding:5px}
</style></head><body>
<header><h1>polarswim</h1><span class="dim" id="sub">loading…</span>
  <span style="flex:1"></span>
  <nav id="tabs">
    <button class="tab on" data-tab="workouts" onclick="tab('workouts')">workouts</button>
    <button class="tab" data-tab="summary" onclick="tab('summary')">summary</button>
    <button class="tab" data-tab="prs" onclick="tab('prs')">personal bests</button>
  </nav>
</header>
<div class="wrap"><aside id="list"></aside><main id="main"></main></div>
<script>
let swims=[],cur=null;
const $=s=>document.querySelector(s);
const fmtTime=s=>s<60?`${Math.round(s)}s`
  :`${Math.floor(s/60)}:${String(Math.round(s%60)).padStart(2,'0')}`;
async function boot(){
  const r=await (await fetch('/api/swims')).json();
  swims=r.swims; $('#sub').textContent=r.summary;
  $('#list').innerHTML=swims.map((s,i)=>
    `<div class="sw" id="sw${i}" onclick="pick(${i})"><b>${s.date}</b>
     <span>${s.yards.toLocaleString()} yd · ${s.n_lengths} lengths</span></div>`).join('');
  if(swims.length) pick(0);
}
async function pick(i){
  cur=swims[i];
  document.querySelectorAll('.sw').forEach(e=>e.classList.remove('on'));
  $('#sw'+i).classList.add('on');
  $('#main').innerHTML='<div class="card dim">loading…</div>';
  const d=await (await fetch('/api/workout/'+cur.id)).json();
  const maxp=Math.max(...d.sets.map(s=>s.pace_s));
  $('#main').innerHTML=`
   <div class="card"><b>${d.header.date}</b> · ${d.header.yards.toLocaleString()} yd ·
     ${d.header.duration} · avg ${d.header.avg_hr??'–'} bpm
     ${d.repairs?`<span class="lo"> · ${d.repairs} turn-detection defect(s) repaired</span>`:''}
     ${d.im_rounds?`<span class="dim"> · ${d.im_rounds} medley round(s)</span>`:''}
     ${d.effort?`<div style="margin-top:10px">
       <span class="chip" style="background:${d.effort.color}">${d.effort.score}</span>
       <span class="dim" style="margin-right:16px">relative load (of 100)</span>
       ${d.effort.intensity!=null?`<span class="chip"
         style="background:${d.effort.intensity_color}">${d.effort.intensity}</span>
       <span class="dim">intensity (of 100)</span>`:''}</div>`:''}</div>
   <div class="card" style="display:flex;gap:22px;align-items:center;flex-wrap:wrap">
     <svg id="pie" width="150" height="150" viewBox="0 0 150 150"></svg>
     <div id="legend" style="font-size:13px"></div></div>
   <div class="card"><svg id="spark" width="100%" height="90"></svg>
     <div class="dim" style="font-size:12px">pace per length — taller is faster</div></div>
   <div class="card"><table><tr><th>set</th><th>reps</th><th>stroke</th><th>conf</th>
     <th>time</th><th>zone</th><th>speed</th><th>pace/25</th><th>rest</th><th></th></tr>
     ${d.sets.map(s=>`<tr><td>${s.set_id}</td>
       <td><b>${s.reps}×${s.rep_yards}</b></td>
       <td class="${s.confidence<0.4?'lo':''}">${s.stroke}</td>
       <td>${s.confidence.toFixed(2)}</td>
       <td>${fmtTime(s.rep_seconds)}</td>
       <td>${s.hr_zone?`<span class="chip" style="background:${s.hr_zone.color}">
           ${s.hr_zone.zone}</span> <span class="dim">${s.hr_zone.pct_max}%</span>`
           :'<span class="dim">–</span>'}</td>
       <td>${s.speed?`<span class="pctbar"><i style="width:${s.speed.percentile}%;
           background:${s.speed.color}"></i></span>
           <span class="dim">${s.speed.percentile}%</span>`
           :'<span class="dim">–</span>'}</td>
       <td><span class="bar" style="width:${Math.round(45*(1-s.pace_s/maxp))+8}px"></span>
           ${s.pace_s.toFixed(0)}s</td>
       <td>${s.rest_before_s.toFixed(0)}s</td>
       <td>${s.pr?'<span class="pr" title="fastest recorded at this distance and stroke">★ PR</span>':''}</td></tr>`).join('')}
   </table></div>
   <div class="card" style="font-size:12px">
     <div class="dim" style="margin-bottom:6px">heart-rate zones — calibrated to your
       observed swim maximum of ${d.hr_max} bpm, which runs below a land-based max</div>
     ${d.zones.map(z=>`<span class="chip" style="background:${z.color}">${z.zone}</span>
       <span style="margin-right:14px">${z.label} ${z.low}–${z.high}</span>`).join('')}
     <div class="dim" style="margin-top:10px">speed — percentile against your own reps
       of the same distance; higher is faster. ★ PR marks your fastest recorded time at
       that distance and stroke (stroke is inferred, so treat it as provisional).</div>
   </div>
   <div class="card"><button onclick="copyCard()">copy Strava card</button>
     <button onclick="downloadImage()">download image for Strava</button>
     <button class="alt" onclick="review()">AI review</button>
     <pre id="card">${d.card.replace(/</g,'&lt;')}</pre></div>
   <div class="card" id="rev" style="display:none"></div>`;
  drawSpark(d.paces);
  drawPie(d.mix);
}
function drawPie(mix){
  const el=$('#pie'); if(!el||!mix||!mix.length)return;
  const cx=75,cy=75,r=62; let a0=-Math.PI/2;
  el.innerHTML=mix.map(m=>{
    const a1=a0+2*Math.PI*m.pct/100;
    const big=(a1-a0)>Math.PI?1:0;
    const x0=cx+r*Math.cos(a0),y0=cy+r*Math.sin(a0);
    const x1=cx+r*Math.cos(a1),y1=cy+r*Math.sin(a1);
    // A single slice covering the whole circle cannot be drawn as an arc path.
    const d=(m.pct>=99.99)
      ? `M ${cx} ${cy-r} A ${r} ${r} 0 1 1 ${cx-0.01} ${cy-r} Z`
      : `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${big} 1 ${x1} ${y1} Z`;
    a0=a1;
    return `<path d="${d}" fill="${m.color}" stroke="#171d24" stroke-width="1.5">
            <title>${m.stroke} ${m.pct.toFixed(0)}%</title></path>`;
  }).join('');
  $('#legend').innerHTML=mix.map(m=>
    `<div style="margin:3px 0"><span style="display:inline-block;width:11px;height:11px;
     background:${m.color};border-radius:2px;margin-right:7px"></span>
     ${m.stroke} <span class="dim">${m.yards} yd · ${m.pct.toFixed(0)}%</span></div>`).join('');
}
function drawSpark(p){
  const el=$('#spark'); if(!el||!p.length)return;
  const w=el.clientWidth||600,h=90,lo=Math.min(...p),hi=Math.max(...p),bw=w/p.length;
  el.innerHTML=p.map((v,i)=>{const f=hi>lo?1-(v-lo)/(hi-lo):.5,bh=Math.max(2,f*(h-6));
    return `<rect x="${i*bw}" y="${h-bh}" width="${Math.max(1,bw-1)}" height="${bh}"
            fill="#4aa3ff" opacity="${.45+.55*f}"><title>#${i+1} ${v.toFixed(1)}s</title></rect>`;
  }).join('');
}
function copyCard(){navigator.clipboard.writeText($('#card').textContent);}
async function downloadImage(){
  // Rasterise the server-rendered SVG through a canvas. No external library, and
  // the SVG uses only <text> and shapes, so the canvas is never tainted.
  const svg=await (await fetch(`/api/image/${cur.id}.svg`)).text();
  const img=new Image();
  const url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml;charset=utf-8'}));
  img.onload=()=>{
    const scale=2;                       // 2x so it stays sharp on a phone
    const c=document.createElement('canvas');
    c.width=img.width*scale; c.height=img.height*scale;
    const ctx=c.getContext('2d');
    ctx.scale(scale,scale);
    ctx.drawImage(img,0,0);
    URL.revokeObjectURL(url);
    c.toBlob(b=>{
      const a=document.createElement('a');
      a.href=URL.createObjectURL(b);
      a.download=`swim-${cur.date.slice(0,10)}.png`;
      a.click();
      setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    },'image/png');
  };
  img.onerror=()=>alert('Could not render the image.');
  img.src=url;
}
async function review(){
  const box=$('#rev'); box.style.display='block';
  box.innerHTML='<span class="dim">asking Claude…</span>';
  const d=await (await fetch('/api/review/'+cur.id)).json();
  box.innerHTML=`<div class="dim" style="font-size:12px">${d.model}</div>
    <div style="white-space:pre-wrap">${d.text.replace(/</g,'&lt;')}</div>`;
}
function tab(name){
  document.querySelectorAll('.tab').forEach(b=>
    b.classList.toggle('on', b.dataset.tab===name));
  document.querySelector('aside').style.display = name==='workouts'?'':'none';
  if(name==='workouts'){ cur?pick(swims.indexOf(cur)):pick(0); }
  else if(name==='summary') loadSummary();
  else loadPRs();
}
async function loadSummary(){
  $('#main').innerHTML='<div class="card dim">loading…</div>';
  const d=await (await fetch('/api/summary')).json();
  const stat=(v,l)=>`<div class="stat"><b>${v}</b><span>${l}</span></div>`;
  $('#main').innerHTML=`
   <div class="card">
     ${stat(d.workouts,'swims')}${stat(d.yards.toLocaleString(),'yards')}
     ${stat(d.hours+'h','pool time')}${stat(d.lengths.toLocaleString(),'lengths')}
     ${stat(d.weeks,'weeks')}
   </div>
   <div class="card"><h2>heart rate</h2>
     ${stat(d.hr_max,'max observed')}${stat(d.hr_p95,'95th percentile')}
     ${stat(d.hr_mean,'mean')}${stat(d.hr_samples.toLocaleString(),'samples')}
     <div style="margin-top:14px">
     ${d.zone_time.map(z=>`<div class="zrow">
       <span class="chip" style="background:${z.color}">${z.zone}</span>
       <span style="width:150px" class="dim">${z.label} ${z.low}–${z.high}</span>
       <span class="ztrack"><i style="width:${z.pct}%;background:${z.color}"></i></span>
       <span style="width:100px" class="dim">${(z.seconds/3600).toFixed(1)}h · ${z.pct}%</span>
     </div>`).join('')}</div>
   </div>
   <div class="card"><h2>training load</h2>
     ${stat(Math.round(d.total_trimp),'total load')}
     ${stat(Math.round(d.mean_trimp),'mean per swim')}
     ${stat(d.longest_yards.toLocaleString(),'longest swim (yd)')}
     <div class="dim" style="font-size:12px;margin-top:8px">
       Banister TRIMP — each second weighted exponentially by heart-rate reserve,
       so time near threshold counts for far more than recovery swimming.</div>
   </div>
   <div class="card"><h2>stroke mix (inferred)</h2>
     ${Object.entries(d.stroke_mix_pct).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
       `<div class="zrow"><span style="width:120px">${k}</span>
        <span class="ztrack"><i style="width:${v}%;background:var(--accent)"></i></span>
        <span style="width:60px" class="dim">${v}%</span></div>`).join('')}
     <div class="dim" style="font-size:12px;margin-top:10px">
       ${d.implausible_reps} reps excluded from personal bests as turn-detection
       artifacts.</div>
   </div>`;
}
const PR_STROKES=['freestyle','backstroke','breaststroke','butterfly','IM','other'];
const PR_LABEL={freestyle:'free',backstroke:'back',breaststroke:'breast',
                butterfly:'fly',IM:'IM',other:'other'};
let prData=null, prStroke='freestyle', prAll=false;

async function loadPRs(){
  $('#main').innerHTML='<div class="card dim">loading…</div>';
  prData=(await (await fetch('/api/prs')).json()).prs;
  // "other" and "undetermined" are both work the classifier could not name, and
  // splitting them across two tabs would imply a distinction that isn't there.
  prData.forEach(p=>{p.tab = PR_STROKES.includes(p.stroke)?p.stroke:'other';});
  if(!prData.some(p=>p.tab===prStroke)) prStroke='freestyle';
  renderPRs();
}
function pickStroke(s){ prStroke=s; renderPRs(); }
function toggleAll(){ prAll=!prAll; renderPRs(); }

function renderPRs(){
  const inTab=prData.filter(p=>p.tab===prStroke);
  const shown=(prAll?inTab:inTab.filter(p=>p.competitive))
                .sort((a,b)=>a.yards-b.yards);
  const hidden=inTab.length-inTab.filter(p=>p.competitive).length;
  const isIM=prStroke==='IM';

  const counts={};
  PR_STROKES.forEach(s=>{counts[s]=prData.filter(p=>p.tab===s&&
    (prAll||p.competitive)).length;});

  $('#main').innerHTML=`
   <div class="card narrow">
     <nav style="margin-bottom:14px">
       ${PR_STROKES.filter(s=>prData.some(p=>p.tab===s)).map(s=>
         `<button class="tab ${s===prStroke?'on':''}" onclick="pickStroke('${s}')">
            ${PR_LABEL[s]} <span class="dim">${counts[s]}</span></button>`).join('')}
     </nav>
     <h2>${PR_LABEL[prStroke]} — personal bests</h2>
     ${shown.length?`<table>
       <tr><th>distance</th><th>best</th><th>pace/25</th>
           ${isIM?'<th>splits <span class="dim">fly · back · breast · free</span></th><th>form</th>':''}
           <th>date</th><th>attempts</th></tr>
       ${shown.map(p=>`<tr>
         <td><b>${p.yards} yd</b></td>
         <td><b>${fmtTime(p.seconds)}</b></td>
         <td class="dim">${p.pace_per_25.toFixed(1)}s</td>
         ${isIM?`<td class="dim">${(p.splits_s||[]).map(x=>x.toFixed(1)).join(' · ')}</td>
                 <td class="dim">${p.form||''}</td>`:''}
         <td class="dim">${p.date}</td>
         <td class="dim">${p.n_attempts}</td></tr>`).join('')}
     </table>`:'<div class="dim">nothing recorded at these distances yet.</div>'}
     <div style="margin-top:14px">
       <button class="alt" onclick="toggleAll()">
         ${prAll?'show racing distances only':`show all distances${hidden?` (+${hidden})`:''}`}
       </button>
     </div>
     <div class="dim" style="font-size:12px;margin-top:12px">
       ${prAll?`Every distance a set happened to be written at, including the 75s and
         125s a practice throws off.`
        :`Distances this stroke is actually raced at, short course yards.`}
       Stroke is inferred from pace, heart rate and rest, not measured, so a best is
       provisional. Reps faster than 65% of the median pace are excluded as
       turn-detection artifacts.
       ${isIM?`<br>A medley is recognised by its repeating four-part shape, so its
         strokes are known rather than inferred. <b>continuous</b> was swum unbroken;
         <b>broken</b> was four reps off the wall, which is the quicker of the two.`:''}
     </div>
   </div>`;
}
async function loadReport(){
  const q=new URLSearchParams();
  if($('#from').value)q.set('from',$('#from').value);
  if($('#to').value)q.set('to',$('#to').value);
  const d=await (await fetch('/api/report?'+q)).json();
  $('#main').innerHTML=`<div class="card"><pre>${d.text}</pre></div>`;
}
boot();
</script></body></html>"""


def create_app(db_url=None) -> Flask:
    app = Flask(__name__)
    engine = db.connect(db_url)

    _ref_cache: dict = {}

    def _reference():
        """Swimmer reference over the whole history. Cached — it scans every length,
        and only changes when a sync adds workouts."""
        n = db.summary(engine)["lengths"]
        if _ref_cache.get("n") != n:
            _ref_cache["ref"] = metrics.build_reference(
                engine, report.classified_lengths(engine))
            _ref_cache["n"] = n
        return _ref_cache["ref"]

    def _fmt(sec):
        m, s = divmod(int(sec or 0), 60)
        return f"{m}:{s:02d}"

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/swims")
    def swims():
        heads = report.workout_headers(engine)
        s = db.summary(engine)
        rows = [{"id": int(r.id), "date": r.start_time[:16],
                 "yards": round((r.distance_m or 0) / 0.9144),
                 "n_lengths": int(r.n_lengths)}
                for r in heads.iloc[::-1].itertuples()]
        return jsonify(swims=rows,
                       summary=f"{s['pool_swims']} swims · {s['lengths']:,} lengths "
                               f"· {s['hr_samples']:,} HR samples")

    @app.get("/api/workout/<int:wid>")
    def workout(wid: int):
        df = report.classified_lengths(engine, wid)
        if df.empty:
            return jsonify(error="not found"), 404
        res = analyze.analyze(engine, workout_id=wid, persist=False)
        repairs = {(r.workout_id, r.idx) for r in res.repairs}
        head = _header(wid)
        ref = _reference()
        return jsonify(
            hr_max=ref.hr_max,
            zones=ref.zone_bounds(),
            effort=ref.effort_score(wid),
            header={"date": head["start_time"][:16],
                    "yards": round((head["distance_m"] or 0) / 0.9144),
                    "duration": _fmt(head["duration_s"]), "avg_hr": head["avg_hr"]},
            sets=report.sets_for_workout(df, repairs, ref),
            mix=[{"stroke": k, "lengths": n, "pct": pct, "yards": n * 25,
                  "color": PIE_COLORS.get(k, "#6b7280")}
                 for k, n, pct in render.stroke_mix(df)],
            paces=[float(x) for x in df.sort_values("idx")["pace_s"]],
            repairs=len(res.repairs),
            im_rounds=len(res.im_rounds),
            card=render.strava_block(df, head))

    @app.get("/api/image/<int:wid>.svg")
    def workout_image(wid: int):
        df = report.classified_lengths(engine, wid)
        if df.empty:
            return jsonify(error="not found"), 404
        res = analyze.analyze(engine, workout_id=wid, persist=False)
        ref = _reference()
        sets = report.sets_for_workout(
            df, {(r.workout_id, r.idx) for r in res.repairs}, ref)
        mix = [{"stroke": k, "pct": pct, "yards": n * 25,
                "color": PIE_COLORS.get(k, "#6b7280")}
               for k, n, pct in render.stroke_mix(df)]
        svg = image.workout_svg(_header(wid), sets, mix, ref.zone_bounds(),
                                ref.hr_max, df)
        return Response(svg, mimetype="image/svg+xml")

    @app.get("/api/review/<int:wid>")
    def review(wid: int):
        df = report.classified_lengths(engine, wid)
        if df.empty:
            return jsonify(error="not found"), 404
        res = analyze.analyze(engine, workout_id=wid, persist=False)
        sets = report.sets_for_workout(df, {(r.workout_id, r.idx) for r in res.repairs})
        try:
            out = ai.review(_header(wid), sets, res.params)
            return jsonify(text=out.text, model=out.model)
        except ai.AIError as e:
            return jsonify(text=f"AI review unavailable: {e}", model="error")

    @app.get("/api/summary")
    def summary_tab():
        return jsonify(report.overall_summary(engine, _reference()))

    @app.get("/api/prs")
    def prs_tab():
        return jsonify(prs=report.personal_bests(engine, _reference()))

    @app.get("/api/report")
    def rep():
        parse = lambda k: (dt.date.fromisoformat(request.args[k])
                           if request.args.get(k) else None)
        s = report.season_summary(engine, parse("from"), parse("to"))
        return jsonify(text=report.format_season(s), data=s)

    def _header(wid: int) -> dict:
        import sqlalchemy as sa
        from .models import workouts
        with engine.connect() as c:
            return dict(c.execute(sa.select(workouts)
                                  .where(workouts.c.id == wid)).mappings().first())

    return app


def serve(db_url=None, port: int = 8770) -> None:
    app = create_app(db_url)
    print(f"polarswim UI  ->  http://127.0.0.1:{port}   (ctrl-c to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
