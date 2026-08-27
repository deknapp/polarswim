"""Local web UI: browse swims, see the analysis, copy the Strava card.

Flask serving one self-contained page — no CDN, no build step, no external
requests — plus a small JSON API the page calls. It binds to localhost only; this
is a personal tool over a personal database, not a service.

The heavy lifting all lives in `analyze` / `report` / `render`, so the UI is a thin
presentation layer over the same derivations the CLI uses.
"""

from __future__ import annotations

import datetime as dt

from flask import Flask, jsonify, request

from . import ai, analyze, db, render, report

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
 .bar{height:9px;background:var(--accent);border-radius:2px;display:inline-block}
 input[type=date]{background:var(--panel);color:var(--fg);border:1px solid var(--line);
        border-radius:5px;padding:5px}
</style></head><body>
<header><h1>polarswim</h1><span class="dim" id="sub">loading…</span>
  <span style="flex:1"></span>
  <input type="date" id="from"><input type="date" id="to">
  <button class="alt" onclick="loadReport()">report</button>
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
     ${d.repairs?`<span class="lo"> · ${d.repairs} merged length(s) repaired</span>`:''}</div>
   <div class="card" style="display:flex;gap:22px;align-items:center;flex-wrap:wrap">
     <svg id="pie" width="150" height="150" viewBox="0 0 150 150"></svg>
     <div id="legend" style="font-size:13px"></div></div>
   <div class="card"><svg id="spark" width="100%" height="90"></svg>
     <div class="dim" style="font-size:12px">pace per length — taller is faster</div></div>
   <div class="card"><table><tr><th>set</th><th>reps</th><th>stroke</th><th>conf</th>
     <th>time</th><th>pace/25</th><th>HR+</th><th>rest</th></tr>
     ${d.sets.map(s=>`<tr><td>${s.set_id}</td>
       <td><b>${s.reps}×${s.rep_yards}</b></td>
       <td class="${s.confidence<0.4?'lo':''}">${s.stroke}</td>
       <td>${s.confidence.toFixed(2)}</td>
       <td>${fmtTime(s.rep_seconds)}</td>
       <td><span class="bar" style="width:${Math.round(50*(1-s.pace_s/maxp))+10}px"></span>
           ${s.pace_s.toFixed(0)}s</td>
       <td>+${s.hr_cost.toFixed(0)}</td><td>${s.rest_before_s.toFixed(0)}s</td></tr>`).join('')}
   </table></div>
   <div class="card"><button onclick="copyCard()">copy Strava card</button>
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
async function review(){
  const box=$('#rev'); box.style.display='block';
  box.innerHTML='<span class="dim">asking Claude…</span>';
  const d=await (await fetch('/api/review/'+cur.id)).json();
  box.innerHTML=`<div class="dim" style="font-size:12px">${d.model}</div>
    <div style="white-space:pre-wrap">${d.text.replace(/</g,'&lt;')}</div>`;
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
        return jsonify(
            header={"date": head["start_time"][:16],
                    "yards": round((head["distance_m"] or 0) / 0.9144),
                    "duration": _fmt(head["duration_s"]), "avg_hr": head["avg_hr"]},
            sets=report.sets_for_workout(df, repairs),
            mix=[{"stroke": k, "lengths": n, "pct": pct, "yards": n * 25,
                  "color": PIE_COLORS.get(k, "#6b7280")}
                 for k, n, pct in render.stroke_mix(df)],
            paces=[float(x) for x in df.sort_values("idx")["pace_s"]],
            repairs=len(res.repairs),
            card=render.strava_block(df, head))

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
