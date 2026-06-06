"""
RAGAS Evaluation Analysis Script
Usage: python ragas_analysis.py --csv path/to/consolidated_eval_results.csv
Outputs: ragas_report.html
"""

import argparse
import csv
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", default="ragas_report_new.html")
args = parser.parse_args()

if not os.path.exists(args.csv):
    print(f"[ERROR] File not found: {args.csv}"); sys.exit(1)

def _f(row, key):
    v = row.get(key, "").strip()
    try: return float(v) if v else None
    except: return None

rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                "id":           row.get("id","").strip(),
                "question":     row.get("question","").strip(),
                "difficulty":   row.get("difficulty","unknown").strip().lower(),
                "category":     row.get("category","unknown").strip().lower(),
                "faithfulness":       _f(row,"faithfulness"),
                "answer_relevancy":   _f(row,"answer_relevancy"),
                "answer_correctness": _f(row,"answer_correctness"),
                "context_precision":  _f(row,"context_precision"),
                "context_recall":     _f(row,"context_recall"),
                "error": row.get("error","").strip(),
            })
        except: continue

total = len(rows)
print(f"[INFO] Loaded {total} rows")

METRICS    = ["faithfulness","answer_relevancy","answer_correctness","context_precision","context_recall"]
BINS       = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
BIN_LABELS = ["0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"]

def get_bin(v):
    if v is None: return -1
    for i in range(len(BINS)-1):
        if BINS[i] <= v < BINS[i+1]: return i
    return len(BINS)-2

def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v)/len(v), 4) if v else None

difficulties = sorted(set(r["difficulty"] for r in rows))
categories   = sorted(set(r["category"]   for r in rows))

diff_counts = {d: sum(1 for r in rows if r["difficulty"]==d) for d in difficulties}
cat_counts  = {c: sum(1 for r in rows if r["category"]==c)   for c in categories}

bin_diff = {str(i): {d: 0 for d in difficulties} for i in range(5)}
for r in rows:
    bi = get_bin(r["answer_correctness"])
    if bi < 0: continue
    bin_diff[str(bi)][r["difficulty"]] += 1

metric_bin_diff = {}
for m in METRICS:
    metric_bin_diff[m] = {str(i): {d: 0 for d in difficulties} for i in range(5)}
    for r in rows:
        bi = get_bin(r[m])
        if bi < 0: continue
        metric_bin_diff[m][str(bi)][r["difficulty"]] += 1

overall        = {m: safe_mean([r[m] for r in rows]) for m in METRICS}
metric_by_diff = {d: {m: safe_mean([r[m] for r in rows if r["difficulty"]==d]) for m in METRICS} for d in difficulties}
cat_correctness_sorted = dict(sorted(
    {c: safe_mean([r["answer_correctness"] for r in rows if r["category"]==c]) for c in categories}.items(),
    key=lambda x: (x[1] or 0)
))

heatmap = {}
for d in difficulties:
    heatmap[d] = {}
    for c in categories:
        subset = [r["answer_correctness"] for r in rows if r["difficulty"]==d and r["category"]==c and r["answer_correctness"] is not None]
        heatmap[d][c] = round(sum(subset)/len(subset), 3) if subset else None

sorted_by_co = sorted([r for r in rows if r["answer_correctness"] is not None], key=lambda r: r["answer_correctness"])
worst50 = sorted_by_co[:50]
top50   = sorted_by_co[-50:][::-1]

def row_summary(r):
    return {"id": r["id"], "q": r["question"][:110]+("..." if len(r["question"])>110 else ""),
            "diff": r["difficulty"], "cat": r["category"], "co": round(r["answer_correctness"],3)}

PALETTE = ["#185FA5","#A32D2D","#BA7517","#3B6D11","#7B3F00",
           "#533FAB","#1A7A4A","#8B0057","#2E6A8E","#5C4033",
           "#4A235A","#1B4F72","#6D4C41","#37474F","#880E4F"]
diff_colors = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(difficulties)}
cat_colors  = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(categories)}

table_rows = [{
    "id": r["id"], "q": r["question"][:110]+("..." if len(r["question"])>110 else ""),
    "diff": r["difficulty"], "cat": r["category"],
    "fa": r["faithfulness"], "re": r["answer_relevancy"],
    "co": r["answer_correctness"], "pr": r["context_precision"], "rc": r["context_recall"],
} for r in rows]

cat_chart_height = max(300, len(categories) * 34)

P = json.dumps({
    "total": total,
    "bin_labels": BIN_LABELS,
    "difficulties": difficulties,
    "categories": categories,
    "metrics": METRICS,
    "diff_colors": diff_colors,
    "cat_colors": cat_colors,
    "diff_counts": diff_counts,
    "cat_counts": cat_counts,
    "bin_diff": bin_diff,
    "metric_bin_diff": metric_bin_diff,
    "overall": overall,
    "metric_by_diff": metric_by_diff,
    "cat_correctness_sorted": cat_correctness_sorted,
    "heatmap": heatmap,
    "worst50": [row_summary(r) for r in worst50],
    "top50":   [row_summary(r) for r in top50],
    "table": table_rows,
})

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAGAS Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f3ef;color:#1a1917;font-size:14px;line-height:1.6}}
.page{{max-width:1140px;margin:0 auto;padding:2rem 1.5rem}}
h1{{font-size:20px;font-weight:700}}
h2{{font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#888;margin:2rem 0 .8rem}}
.meta{{font-size:12px;color:#888;margin-top:3px}}

/* collapsible tag sections */
.tag-section{{margin-top:10px}}
.tag-section-header{{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none;padding:4px 0}}
.tag-section-header:hover .tag-section-title{{color:#1a1917}}
.tag-section-title{{font-size:11px;font-weight:600;letter-spacing:.7px;text-transform:uppercase;color:#aaa;transition:color .15s}}
.tag-section-arrow{{font-size:10px;color:#bbb;transition:transform .2s;display:inline-block}}
.tag-section-arrow.open{{transform:rotate(90deg)}}
.tag-list{{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}}
.tag-list.hidden{{display:none}}
.tag{{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;border:1px solid rgba(0,0,0,0.08);background:#fff}}
.tag-count{{font-weight:700;font-size:11px;color:#555}}

/* metric cards */
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:1rem 0}}
.card{{background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:10px;padding:1rem 1.2rem}}
.lbl{{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:#999}}
.val{{font-size:24px;font-weight:700;margin-top:2px}}

/* layout */
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
.panel{{background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:10px;padding:1.2rem}}
.panel-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.ptitle{{font-size:11px;color:#aaa}}
.dl-btn{{padding:3px 10px;font-size:10px;font-weight:500;border:1px solid rgba(0,0,0,0.1);border-radius:5px;background:#f9f8f5;cursor:pointer;color:#666;font-family:inherit;transition:all .15s}}
.dl-btn:hover{{background:#1a1917;color:#fff;border-color:#1a1917}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;font-size:11px;color:#666;margin-bottom:8px}}
.legend span{{display:flex;align-items:center;gap:4px}}
.dot{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}
.ch{{position:relative;width:100%}}

/* metric tabs */
.metric-tabs{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}}
.mtab{{padding:3px 11px;border-radius:20px;font-size:11px;font-weight:500;border:1px solid rgba(0,0,0,0.1);background:#f4f3ef;cursor:pointer;transition:all .15s;font-family:inherit}}
.mtab.active{{background:#1a1917;color:#fff;border-color:#1a1917}}

/* tables */
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:6px 8px;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:#aaa;border-bottom:1px solid #eee}}
td{{padding:6px 8px;border-bottom:1px solid #f5f5f5;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
.badge{{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}}
.bw{{display:flex;align-items:center;gap:6px}}
.bt{{flex:1;height:5px;background:#eee;border-radius:3px;overflow:hidden;min-width:50px}}
.bf{{height:100%;border-radius:3px}}

/* bottom/top tabs */
.wt-tabs{{display:flex;gap:0;margin-bottom:12px;border-bottom:1px solid #eee}}
.wttab{{padding:6px 16px;font-size:12px;font-weight:500;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;color:#aaa}}
.wttab.active{{color:#1a1917;border-bottom-color:#1a1917}}

/* full table controls */
.ctrl{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:.8rem;align-items:center}}
.ctrl input,.ctrl select{{padding:5px 10px;border:1px solid rgba(0,0,0,0.12);border-radius:6px;font-size:12px;background:#fff;color:#1a1917;font-family:inherit}}
.ctrl input{{width:220px}}

/* pagination */
.pagination{{display:flex;align-items:center;gap:5px;margin-top:10px;flex-wrap:wrap}}
.pgbtn{{padding:4px 10px;border:1px solid rgba(0,0,0,0.12);border-radius:5px;background:#fff;font-size:11px;cursor:pointer;font-family:inherit}}
.pgbtn:disabled{{opacity:.3;cursor:default}}
.pgbtn.active{{background:#1a1917;color:#fff;border-color:#1a1917}}
.pginfo{{font-size:11px;color:#aaa;margin-left:4px}}

/* heatmap */
.heatmap-wrap{{overflow-x:auto}}
.hm-table{{border-collapse:collapse;font-size:11px;min-width:100%}}
.hm-table th{{padding:5px 8px;font-size:9px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:#aaa;border:1px solid #eee;white-space:nowrap}}
.hm-table td{{padding:5px 8px;border:1px solid #eee;text-align:center;font-variant-numeric:tabular-nums;font-size:11px;white-space:nowrap}}
.hm-rowlabel{{text-align:left!important;font-weight:500;color:#555}}

.topbar{{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem}}
.fname{{font-size:11px;color:#ccc;padding-top:4px}}
</style>
</head>
<body>
<div class="page">

<!-- ── Header ── -->
<div class="topbar">
  <div>
    <h1>RAGAS Evaluation Report</h1>
    <p class="meta" id="meta"></p>
  </div>
  <div class="fname">{os.path.basename(args.csv)}</div>
</div>

<!-- ── Overall metrics ── -->
<h2>Overall metrics</h2>
<div class="cards" id="summary"></div>

<!-- ── Collapsible tag sections ── -->
<div class="tag-section">
  <div class="tag-section-header" onclick="toggleSection('diff-list','diff-arrow')">
    <span class="tag-section-arrow open" id="diff-arrow">▶</span>
    <span class="tag-section-title">Difficulty</span>
  </div>
  <div class="tag-list" id="diff-list"></div>
</div>
<div class="tag-section">
  <div class="tag-section-header" onclick="toggleSection('cat-list','cat-arrow')">
    <span class="tag-section-arrow" id="cat-arrow">▶</span>
    <span class="tag-section-title">Category</span>
  </div>
  <div class="tag-list hidden" id="cat-list"></div>
</div>

<!-- ── Correctness distribution + all metrics tabs — 2-col grid ── -->
<h2>Answer correctness &amp; metric distributions — by difficulty</h2>
<div class="grid2">
  <div class="panel">
    <div class="panel-header">
      <span class="ptitle">answer correctness</span>
      <button class="dl-btn" onclick="dlChart('c1','correctness_distribution')">↓ Download</button>
    </div>
    <div class="legend" id="ld"></div>
    <div class="ch" style="height:220px"><canvas id="c1"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-header">
      <span class="ptitle">select metric</span>
      <button class="dl-btn" onclick="dlChart('c_metric','metric_distribution')">↓ Download</button>
    </div>
    <div class="metric-tabs" id="mtabs"></div>
    <div class="legend" id="ld2"></div>
    <div class="ch" style="height:220px"><canvas id="c_metric"></canvas></div>
  </div>
</div>

<!-- ── Avg metrics by difficulty + Avg correctness by category — 2-col grid ── -->
<h2>Metric averages &amp; category breakdown</h2>
<div class="grid2">
  <div class="panel">
    <div class="panel-header">
      <span class="ptitle">avg RAGAS metrics by difficulty</span>
      <button class="dl-btn" onclick="dlChart('c3','avg_metrics_by_difficulty')">↓ Download</button>
    </div>
    <div class="ch" style="height:260px"><canvas id="c3"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-header">
      <span class="ptitle">avg answer correctness by category</span>
      <button class="dl-btn" onclick="dlChart('c_cat','avg_correctness_by_category')">↓ Download</button>
    </div>
    <div class="ch" id="cat-wrap" style="height:{cat_chart_height}px"><canvas id="c_cat"></canvas></div>
  </div>
</div>

<!-- ── Heatmap ── -->
<h2>Difficulty × category heatmap — avg answer correctness</h2>
<div class="panel">
  <div class="panel-header">
    <span class="ptitle">red = poor · amber = moderate · green = good · — = no data</span>
    <button class="dl-btn" onclick="dlHeatmap()">↓ Download</button>
  </div>
  <div class="heatmap-wrap"><table class="hm-table" id="hm-table"></table></div>
</div>

<!-- ── Bottom / Top 50 ── -->
<h2>Bottom 50 / Top 50 — answer correctness</h2>
<div class="panel" style="overflow-x:auto">
  <div class="wt-tabs">
    <div class="wttab active" id="tab-worst" onclick="showWT('worst')">↓ Bottom 50 — lowest</div>
    <div class="wttab" id="tab-top"   onclick="showWT('top')">↑ Top 50 — highest</div>
  </div>
  <table>
    <thead><tr><th>#</th><th>ID</th><th>Question</th><th>Difficulty</th><th>Category</th><th>Correctness</th></tr></thead>
    <tbody id="wt-body"></tbody>
  </table>
</div>

<!-- ── Full metrics table ── -->
<h2>Full metrics table</h2>
<div class="ctrl">
  <input id="search" placeholder="Search ID or question...">
  <select id="fdiff"><option value="">All difficulties</option></select>
  <select id="fcat"><option value="">All categories</option></select>
</div>
<div class="panel" style="overflow-x:auto">
  <table>
    <thead><tr><th>ID</th><th>Question</th><th>Diff</th><th>Cat</th>
      <th>Faithful</th><th>Relevancy</th><th>Correctness</th><th>Precision</th><th>Recall</th></tr></thead>
    <tbody id="fb"></tbody>
  </table>
</div>
<div class="pagination" id="pag"></div>

</div><!-- .page -->
<script src="chart.min.js"></script>
<script>
const D={P};
const ML={{faithfulness:"Faithfulness",answer_relevancy:"Relevancy",
  answer_correctness:"Correctness",context_precision:"Precision",context_recall:"Recall"}};

function fmt(v){{return v!==null&&v!==undefined?Number(v).toFixed(3):"–"}}
function sc(v){{
  if(v===null||v===undefined)return"#999";
  return v>=0.8?"#3B6D11":v>=0.6?"#BA7517":"#A32D2D";
}}
function heatColor(v){{
  if(v===null||v===undefined)return"#f9f9f9";
  if(v>=0.8)return"#d4edda"; if(v>=0.6)return"#fff3cd";
  if(v>=0.4)return"#ffe5cc"; return"#f8d7da";
}}
function bs(val){{
  const m={{easy:"#E6F1FB|#0C447C",hard:"#FCEBEB|#501313",medium:"#FAEEDA|#633806",
    factual:"#E6F1FB|#0C447C",mechanism:"#EAF3DE|#173404",
    analysis:"#F3E8FF|#3B0764",comparison:"#FFF3E0|#4E342E",
    complex_mechanism:"#FCE4EC|#4A0010",complex_reasoning:"#E8F5E9|#1B5E20",
    conceptual:"#E3F2FD|#0D47A1",critical_analysis:"#FFF8E1|#3E2723",
    critical_appraisal:"#F9FBE7|#33691E",definition:"#EDE7F6|#1A237E",
    methodology:"#E0F7FA|#006064"}};
  const[bg,tx]=(m[val]||"#eee|#333").split("|");
  return`background:${{bg}};color:${{tx}}`;
}}

// ── collapsible sections ──────────────────────────────────────────────────────
function toggleSection(listId, arrowId){{
  const list=document.getElementById(listId);
  const arrow=document.getElementById(arrowId);
  const hidden=list.classList.toggle("hidden");
  arrow.classList.toggle("open",!hidden);
}}

// ── meta ──────────────────────────────────────────────────────────────────────
document.getElementById("meta").textContent=`${{D.total.toLocaleString()}} questions`;

// difficulty tags
const dl=document.getElementById("diff-list");
D.difficulties.forEach(d=>{{
  dl.innerHTML+=`<span class="tag"><span class="dot" style="background:${{D.diff_colors[d]}}"></span>${{d}}<span class="tag-count">${{D.diff_counts[d]}}</span></span>`;
}});

// category tags
const cl=document.getElementById("cat-list");
D.categories.forEach(c=>{{
  cl.innerHTML+=`<span class="tag"><span class="dot" style="background:${{D.cat_colors[c]}}"></span>${{c}}<span class="tag-count">${{D.cat_counts[c]}}</span></span>`;
}});

// ── summary cards ─────────────────────────────────────────────────────────────
const sm=document.getElementById("summary");
D.metrics.forEach(m=>{{
  const v=D.overall[m];
  sm.innerHTML+=`<div class="card"><div class="lbl">${{ML[m]}}</div>
    <div class="val" style="color:${{sc(v)}}">${{fmt(v)}}</div></div>`;
}});

// legend builder
function leg(id,keys,cm){{
  const el=document.getElementById(id);
  if(!el)return;
  keys.forEach(k=>{{el.innerHTML+=`<span><span class="dot" style="background:${{cm[k]}}"></span>${{k}}</span>`;}});
}}
leg("ld", D.difficulties, D.diff_colors);
leg("ld2",D.difficulties, D.diff_colors);

Chart.defaults.font.size=11;
Chart.defaults.color="#aaa";

// ── download chart helper ─────────────────────────────────────────────────────
function dlChart(canvasId, filename){{
  const canvas=document.getElementById(canvasId);
  const tmp=document.createElement("canvas");
  tmp.width=canvas.width; tmp.height=canvas.height;
  const ctx=tmp.getContext("2d");
  ctx.fillStyle="#ffffff";
  ctx.fillRect(0,0,tmp.width,tmp.height);
  ctx.drawImage(canvas,0,0);
  const a=document.createElement("a");
  a.href=tmp.toDataURL("image/png");
  a.download=filename+".png";
  a.click();
}}

function dlHeatmap(){{
  const tbl=document.getElementById("hm-table");
  const canvas=document.createElement("canvas");
  const rows=tbl.rows;
  const cols=rows[0].cells.length;
  const cw=110, ch=34, pad=10;
  canvas.width=cols*cw+pad*2;
  canvas.height=rows.length*ch+pad*2;
  const ctx=canvas.getContext("2d");
  ctx.fillStyle="#ffffff"; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.font="11px -apple-system,sans-serif";
  for(let r=0;r<rows.length;r++){{
    for(let c=0;c<rows[r].cells.length;c++){{
      const cell=rows[r].cells[c];
      const x=pad+c*cw, y=pad+r*ch;
      ctx.fillStyle=cell.style.backgroundColor||"#f9f9f9";
      ctx.fillRect(x,y,cw,ch);
      ctx.strokeStyle="#eee"; ctx.strokeRect(x,y,cw,ch);
      ctx.fillStyle=cell.style.color||"#333";
      ctx.fillText(cell.textContent.trim(),x+6,y+ch/2+4);
    }}
  }}
  const a=document.createElement("a");
  a.href=canvas.toDataURL("image/png");
  a.download="heatmap_difficulty_category.png";
  a.click();
}}

// ── tooltip: show all breakdowns + total ──────────────────────────────────────
function binTooltip(){{
  return{{callbacks:{{
    title:ctx=>ctx[0].label,
    afterBody:ctx=>{{
      const bi=ctx[0].dataIndex;
      let total=0;
      const lines=ctx[0].chart.data.datasets.map(ds=>{{
        const v=ds.data[bi]||0; total+=v;
        return`  ${{ds.label}}: ${{v}}`;
      }});
      lines.push(`  ─────────`);
      lines.push(`  total: ${{total}}`);
      return lines;
    }},
    label:()=>null,
  }}}};
}}

function binDatasets(binData){{
  return D.difficulties.map(k=>({{
    label:k,
    data:D.bin_labels.map((_,i)=>(binData[String(i)][k]||0)),
    backgroundColor:D.diff_colors[k],
    borderRadius:3,
  }}));
}}

// ── Chart 1: answer correctness distribution ──────────────────────────────────
new Chart(document.getElementById("c1"),{{
  type:"bar",
  data:{{labels:D.bin_labels,datasets:binDatasets(D.bin_diff)}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:binTooltip()}},
    scales:{{x:{{stacked:true,grid:{{display:false}}}},
             y:{{stacked:true,beginAtZero:true,ticks:{{precision:0}},grid:{{color:"rgba(0,0,0,0.04)"}}}}}}
  }}
}});

// ── Chart: metric distribution tabs ──────────────────────────────────────────
let metricChartInst=null;
let activeMetric=D.metrics[0];
function drawMetricChart(metric){{
  if(metricChartInst) metricChartInst.destroy();
  metricChartInst=new Chart(document.getElementById("c_metric"),{{
    type:"bar",
    data:{{labels:D.bin_labels,datasets:binDatasets(D.metric_bin_diff[metric])}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:binTooltip()}},
      scales:{{x:{{stacked:true,grid:{{display:false}}}},
               y:{{stacked:true,beginAtZero:true,ticks:{{precision:0}},grid:{{color:"rgba(0,0,0,0.04)"}}}}}}
    }}
  }});
}}
const tabsEl=document.getElementById("mtabs");
D.metrics.forEach(m=>{{
  const btn=document.createElement("button");
  btn.className="mtab"+(m===activeMetric?" active":"");
  btn.textContent=ML[m];
  btn.onclick=()=>{{
    document.querySelectorAll(".mtab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    activeMetric=m; drawMetricChart(m);
  }};
  tabsEl.appendChild(btn);
}});
drawMetricChart(activeMetric);

// ── Chart 3: avg metrics by difficulty ───────────────────────────────────────
new Chart(document.getElementById("c3"),{{
  type:"bar",
  data:{{
    labels:D.metrics.map(m=>ML[m]),
    datasets:D.difficulties.map(d=>({{
      label:d,
      data:D.metrics.map(m=>D.metric_by_diff[d][m]??0),
      backgroundColor:D.diff_colors[d],
      borderRadius:3,
    }}))
  }},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:"bottom",labels:{{boxWidth:10,font:{{size:11}}}}}}}},
    scales:{{x:{{grid:{{display:false}}}},
             y:{{min:0,max:1,ticks:{{stepSize:0.2}},grid:{{color:"rgba(0,0,0,0.04)"}}}}}}
  }}
}});

// ── Chart: avg correctness per category (horizontal) ─────────────────────────
const catKeys=Object.keys(D.cat_correctness_sorted);
const catVals=catKeys.map(c=>D.cat_correctness_sorted[c]??0);
new Chart(document.getElementById("c_cat"),{{
  type:"bar",
  data:{{
    labels:catKeys,
    datasets:[{{
      label:"avg correctness",
      data:catVals,
      backgroundColor:catVals.map(v=>v>=0.8?"#3B6D11":v>=0.6?"#BA7517":"#A32D2D"),
      borderRadius:3,
    }}]
  }},
  options:{{
    indexAxis:"y",
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`avg correctness: ${{ctx.parsed.x.toFixed(3)}}`}}}}}},
    scales:{{
      x:{{min:0,max:1,ticks:{{stepSize:0.2}},grid:{{color:"rgba(0,0,0,0.04)"}}}},
      y:{{grid:{{display:false}}}}
    }}
  }}
}});

// ── Heatmap ───────────────────────────────────────────────────────────────────
(function(){{
  const tbl=document.getElementById("hm-table");
  let h=`<tr><th></th>${{D.categories.map(c=>`<th>${{c}}</th>`).join("")}}</tr>`;
  D.difficulties.forEach(d=>{{
    h+=`<tr><td class="hm-rowlabel">${{d}}</td>`;
    D.categories.forEach(c=>{{
      const v=(D.heatmap[d]&&D.heatmap[d][c]!==null&&D.heatmap[d][c]!==undefined)?D.heatmap[d][c]:null;
      h+=`<td style="background:${{heatColor(v)}};color:${{v!==null?sc(v):"#ccc"}}">${{v!==null?v.toFixed(3):"–"}}</td>`;
    }});
    h+=`</tr>`;
  }});
  tbl.innerHTML=h;
}})();

// ── Bottom / Top 50 ───────────────────────────────────────────────────────────
let wtMode="worst";
function showWT(mode){{
  wtMode=mode;
  document.getElementById("tab-worst").classList.toggle("active",mode==="worst");
  document.getElementById("tab-top").classList.toggle("active",mode==="top");
  renderWT();
}}
function renderWT(){{
  const data=wtMode==="worst"?D.worst50:D.top50;
  const tb=document.getElementById("wt-body");
  tb.innerHTML="";
  data.forEach((r,i)=>{{
    const c=sc(r.co);
    tb.innerHTML+=`<tr>
      <td style="color:#ccc;width:28px">${{i+1}}</td>
      <td style="font-size:11px;white-space:nowrap">${{r.id}}</td>
      <td style="max-width:300px;color:#555">${{r.q}}</td>
      <td><span class="badge" style="${{bs(r.diff)}}">${{r.diff}}</span></td>
      <td><span class="badge" style="${{bs(r.cat)}}">${{r.cat}}</span></td>
      <td style="min-width:150px"><div class="bw">
        <span style="color:${{c}};min-width:42px;font-variant-numeric:tabular-nums">${{r.co.toFixed(3)}}</span>
        <div class="bt"><div class="bf" style="width:${{(Math.max(0,r.co)*100).toFixed(1)}}%;background:${{c}}"></div></div>
      </div></td>
    </tr>`;
  }});
}}
renderWT();

// ── Full table with pagination ────────────────────────────────────────────────
D.difficulties.forEach(d=>{{document.getElementById("fdiff").innerHTML+=`<option value="${{d}}">${{d}}</option>`;}});
D.categories.forEach(c=>{{document.getElementById("fcat").innerHTML+=`<option value="${{c}}">${{c}}</option>`;}});

const PAGE_SIZE=50;
let currentPage=0;
let filteredRows=[];

function applyFilters(){{
  const q=document.getElementById("search").value.toLowerCase();
  const df=document.getElementById("fdiff").value;
  const cf=document.getElementById("fcat").value;
  filteredRows=D.table.filter(r=>
    (!q||r.id.toLowerCase().includes(q)||r.q.toLowerCase().includes(q))&&
    (!df||r.diff===df)&&(!cf||r.cat===cf));
  currentPage=0; renderTable();
}}

function renderTable(){{
  const start=currentPage*PAGE_SIZE;
  const fb=document.getElementById("fb");
  fb.innerHTML="";
  filteredRows.slice(start,start+PAGE_SIZE).forEach(r=>{{
    fb.innerHTML+=`<tr>
      <td style="font-size:11px;white-space:nowrap">${{r.id}}</td>
      <td style="max-width:240px;color:#555">${{r.q}}</td>
      <td><span class="badge" style="${{bs(r.diff)}}">${{r.diff}}</span></td>
      <td><span class="badge" style="${{bs(r.cat)}}">${{r.cat}}</span></td>
      <td style="color:${{sc(r.fa)}};font-variant-numeric:tabular-nums">${{fmt(r.fa)}}</td>
      <td style="color:${{sc(r.re)}};font-variant-numeric:tabular-nums">${{fmt(r.re)}}</td>
      <td style="color:${{sc(r.co)}};font-variant-numeric:tabular-nums">${{fmt(r.co)}}</td>
      <td style="color:${{sc(r.pr)}};font-variant-numeric:tabular-nums">${{fmt(r.pr)}}</td>
      <td style="color:${{sc(r.rc)}};font-variant-numeric:tabular-nums">${{fmt(r.rc)}}</td>
    </tr>`;
  }});
  renderPagination();
}}

function renderPagination(){{
  const totalPages=Math.ceil(filteredRows.length/PAGE_SIZE);
  const pag=document.getElementById("pag");
  pag.innerHTML="";
  const mk=(label,page,disabled,active)=>{{
    const b=document.createElement("button");
    b.className="pgbtn"+(active?" active":"");
    b.textContent=label; b.disabled=disabled;
    b.onclick=()=>{{
      currentPage=page; renderTable();
      document.getElementById("fb").scrollIntoView({{behavior:"smooth",block:"start"}});
    }};
    return b;
  }};
  pag.appendChild(mk("« First",0,currentPage===0,false));
  pag.appendChild(mk("‹ Prev",currentPage-1,currentPage===0,false));
  const s=Math.max(0,currentPage-2), e=Math.min(totalPages,s+5);
  for(let i=s;i<e;i++) pag.appendChild(mk(String(i+1),i,false,i===currentPage));
  pag.appendChild(mk("Next ›",currentPage+1,currentPage>=totalPages-1,false));
  pag.appendChild(mk("Last »",totalPages-1,currentPage>=totalPages-1,false));
  const info=document.createElement("span");
  info.className="pginfo";
  const s2=currentPage*PAGE_SIZE+1, e2=Math.min((currentPage+1)*PAGE_SIZE,filteredRows.length);
  info.textContent=`${{s2}}–${{e2}} of ${{filteredRows.length}} rows`;
  pag.appendChild(info);
}}

document.getElementById("search").addEventListener("input",applyFilters);
document.getElementById("fdiff").addEventListener("change",applyFilters);
document.getElementById("fcat").addEventListener("change",applyFilters);
applyFilters();
</script>
</body>
</html>"""

with open(args.out,"w",encoding="utf-8") as f:
    f.write(html)
print(f"[DONE] Report saved -> {args.out}")
print(f"       Open: file://{os.path.abspath(args.out)}")