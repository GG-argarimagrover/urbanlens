#!/usr/bin/env python3
"""Build a self-contained interactive dashboard (dashboard.html) from an UrbanLens output folder.

    python dashboard.py out_demo          # -> out_demo/dashboard.html
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

CELL_COLS = ["cell_id", "pop", "deprivation", "elderly_share", "overall", "walk", "safety", "comfort", "transit",
             "green", "heat", "heat_vulnerability", "heat_exposure", "need", "priority", "is_priority", "typology",
             "lst_c", "ndvi", "veg_frac", "park_dist_m", "pedshed", "int_density", "dest_cats", "ptal_ai",
             "sidewalk_share", "junc_risk"]
JUNC_COLS = ["node", "vmax_kmh", "lanes", "legs", "crossing", "risk", "demand", "priority", "unprotected"]


def _clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 2)
    if v is pd.NA or v is None:
        return None
    return v


def _fc(gdf, cols, prec=1e-5):
    gdf = gdf.copy()
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, prec)
    feats = []
    for _, r in gdf.iterrows():
        feats.append({"type": "Feature", "properties": {c: _clean(r[c]) for c in cols if c in gdf.columns},
                      "geometry": json.loads(shapely.to_geojson(r.geometry))})
    return {"type": "FeatureCollection", "features": feats}


def build_dashboard(out_dir, path=None):
    p = lambda n: os.path.join(out_dir, n)
    cells = gpd.read_file(p("cells.geojson"))
    junc = gpd.read_file(p("junctions.geojson"))
    summary = json.load(open(p("summary.json")))
    recs = pd.read_csv(p("recommendations.csv")) if os.path.exists(p("recommendations.csv")) else pd.DataFrame()
    zones = gpd.read_file(p("zones.geojson")) if os.path.exists(p("zones.geojson")) else None

    junc = junc.sort_values("priority", ascending=False)
    junc["lat"], junc["lon"] = junc.geometry.y.round(5), junc.geometry.x.round(5)
    data = {
        "summary": summary,
        "cells": _fc(cells, CELL_COLS),
        "zones": _fc(zones, [c for c in zones.columns if c != "geometry"]) if zones is not None else
                 {"type": "FeatureCollection", "features": []},
        "junctions": [{**{c: _clean(r[c]) for c in JUNC_COLS if c in junc.columns}, "lat": r.lat, "lon": r.lon}
                      for _, r in junc.iterrows()],
        "recs": [{k: _clean(v) for k, v in r.items()} for r in recs.to_dict("records")],
    }
    blob = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    path = path or p("dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  · dashboard -> {path} ({len(html)/1e6:.1f} MB)")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UrbanLens dashboard</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--bg:#f6f7f9;--panel:#fff;--fg:#17191c;--mut:#667;--line:#e2e4e8;--acc:#0b6e4f;--chip:#eef0f3;--shadow:0 1px 3px rgba(0,0,0,.12)}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--panel:#1b1e23;--fg:#ecedef;--mut:#99a;--line:#30343b;--acc:#4cc38a;--chip:#272b32;--shadow:0 1px 3px rgba(0,0,0,.5)}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px 14px 4px}h1{font-size:18px;margin:0}.sub{color:var(--mut);font-size:12px}
.warn{background:#fff3cd;color:#664d03;padding:6px 10px;border-radius:8px;margin:8px 14px 0;font-size:12px}
.kpis{display:flex;gap:8px;overflow-x:auto;padding:10px 14px;scrollbar-width:none}.kpis::-webkit-scrollbar{display:none}
.kpi{flex:0 0 118px;background:var(--panel);border-radius:12px;padding:9px 10px;box-shadow:var(--shadow);cursor:pointer;border:2px solid transparent}
.kpi.on{border-color:var(--acc)}.kpi .l{font-size:11px;color:var(--mut);height:28px;line-height:1.25}.kpi .v{font-size:26px;font-weight:700}
.bar{height:5px;background:var(--line);border-radius:3px}.bar i{display:block;height:5px;border-radius:3px}
nav{display:flex;gap:6px;overflow-x:auto;padding:4px 14px 8px;position:sticky;top:0;background:var(--bg);z-index:900;scrollbar-width:none}
nav button{border:0;background:var(--chip);color:var(--fg);padding:8px 14px;border-radius:20px;font:inherit;white-space:nowrap;cursor:pointer}
nav button.on{background:var(--acc);color:#fff}
main{padding:0 14px 40px;max-width:1180px;margin:0 auto}section{display:none}section.on{display:block}
.panel{background:var(--panel);border-radius:12px;padding:12px;box-shadow:var(--shadow);margin-bottom:12px}
.panel h3{margin:0 0 8px;font-size:14px}.grid2{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:860px){.grid2{grid-template-columns:2fr 1fr}.kpi{flex:1 0 118px}}
#map{height:56vh;min-height:340px;border-radius:12px}.leaflet-container{background:#dfe3e8}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px}.chips button{border:0;background:var(--chip);color:var(--fg);padding:5px 10px;border-radius:14px;font:inherit;font-size:12px;cursor:pointer}
.chips button.on{background:var(--acc);color:#fff}
.legend{height:10px;border-radius:5px;margin:6px 0 2px}.lg{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}
.row{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px solid var(--line);font-size:13px}.row:last-child{border:0}
.row b{font-weight:600}table{width:100%;border-collapse:collapse}th,td{padding:6px 6px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}th{color:var(--mut);font-weight:500;font-size:12px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;color:#fff}.tag{font-size:11px;background:var(--chip);padding:1px 7px;border-radius:9px;color:var(--mut)}
details{border-top:1px solid var(--line);padding:7px 0}summary{cursor:pointer;font-weight:600}details p{margin:6px 0 2px}small,.mut{color:var(--mut)}
.btn{border:0;background:var(--acc);color:#fff;padding:6px 12px;border-radius:8px;font:inherit;font-size:12px;cursor:pointer}
.ins li{margin:4px 0}.cv{position:relative;height:260px}input[type=range]{width:100%}
pre{white-space:pre-wrap;font:inherit;margin:0}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="sub"></div></header>
<div id="warn"></div>
<div class="kpis" id="kpis"></div>
<nav id="tabs"></nav>
<main>
<section id="t-map" class="on">
 <div class="grid2"><div>
  <div class="chips" id="ind"></div>
  <div id="map"></div>
  <div class="legend" id="legend"></div><div class="lg"><span id="lg0"></span><span id="lg1"></span></div>
  <div class="chips" style="margin-top:8px"><label><input type="checkbox" id="oZones" checked> Priority zones</label>
   <label><input type="checkbox" id="oJunc"> Junction risk</label><label><input type="checkbox" id="oBase" checked> Basemap</label></div>
 </div>
 <div><div class="panel" id="cellPanel"><h3>Tap a cell</h3><div class="mut">Select any grid cell to see its indicators, typology and the evidence behind its score.</div></div>
  <div class="panel"><h3>Key findings</h3><ul class="ins" id="ins" style="padding-left:18px;margin:0"></ul></div></div></div>
</section>
<section id="t-eq">
 <div class="chips" id="eqInd"></div>
 <div class="grid2"><div class="panel"><h3 id="eqT">Score by deprivation quintile</h3><div class="cv"><canvas id="cEq"></canvas></div>
   <small>Q1 = least deprived · Q5 = most deprived (population-weighted means).</small></div>
  <div class="panel"><h3>Profile: most vs least deprived</h3><div class="cv"><canvas id="cRad"></canvas></div></div></div>
 <div class="panel"><h3>Equity gaps</h3><div style="overflow-x:auto"><table id="eqTab"></table></div>
  <p><small>Concentration index &gt; 0: benefits concentrated among the least-deprived; &lt; 0: among the most-deprived.</small></p></div>
</section>
<section id="t-zones"><div id="zones"></div></section>
<section id="t-j">
 <div class="panel"><h3>Junction risk screen</h3><div>Show junctions with risk ≥ <b id="rv">45</b> <span class="mut">(<span id="rc"></span> junctions)</span></div>
  <input type="range" id="rs" min="0" max="90" value="45"><small>Heuristic screen: speed→injury severity × width × legs × crossing/sidewalk mitigation. Not a crash model.</small></div>
 <div class="panel"><h3>Top 15 junctions by risk × pedestrian demand</h3><div style="overflow-x:auto"><table id="jt"></table></div></div>
</section>
<section id="t-ty">
 <div class="grid2"><div class="panel"><h3>Population by AI typology</h3><div class="cv"><canvas id="cTy"></canvas></div></div>
 <div class="panel"><h3>Typologies</h3><table id="tyTab"></table></div></div>
 <div class="panel"><h3>Planning brief <small id="bsrc"></small></h3><pre id="brief"></pre></div>
</section>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D = __DATA__;
const S = D.summary, F = D.cells.features;
const $ = s => document.querySelector(s);
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SC = ['walk','safety','comfort','transit','green','heat'];
const LBL = {overall:'Overall',walk:'Walkability',safety:'Junction safety',comfort:'Pedestrian comfort',transit:'Public transport',green:'Green access',heat:'Heat resilience',heat_vulnerability:'Climate vulnerability',need:'Social need',priority:'Priority index'};
const SEQ = new Set(['heat_vulnerability','need','priority']);
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
let cur = 'overall', selected = null, eqKey = 'overall';

// ---------- colour ramps ----------
const hex = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16));
function ramp(stops, t){ t = Math.max(0, Math.min(1, t)); const x = t*(stops.length-1), i = Math.min(stops.length-2, Math.floor(x)); const a = hex(stops[i]), b = hex(stops[i+1]), f = x-i;
  return 'rgb(' + a.map((v,k) => Math.round(v + (b[k]-v)*f)).join(',') + ')'; }
const DIV = ['#b2182b','#ef8a62','#fddbc7','#d9f0d3','#7fbf7b','#1b7837'], SQ = ['#ffffcc','#fd8d3c','#800026'];
const range = k => { const v = F.map(f => f.properties[k]).filter(x => x != null); return [Math.min(...v), Math.max(...v)]; };
const col = (k, v) => v == null ? '#999' : SEQ.has(k) ? ramp(SQ, (v - R[k][0]) / ((R[k][1] - R[k][0]) || 1)) : ramp(DIV, v / 100);
const R = {}; ['heat_vulnerability','need','priority'].forEach(k => R[k] = range(k));
const scol = v => ramp(DIV, v / 100);

// ---------- header, KPIs, tabs ----------
$('#title').textContent = S.city; $('#sub').textContent = `${S.cells} analysis cells · ${Math.round(S.population).toLocaleString()} residents modelled`;
const NOEQ = (S.provenance.socio||'').includes('PROXY');
if (NOEQ) $('#t-eq').insertAdjacentHTML('afterbegin', '<div class="warn" style="margin:0 0 12px">⚠ No deprivation data was supplied, so every area is treated as equally deprived. The equity charts below are NOT meaningful — rerun with <code>--socio</code> data.</div>');
if (S.provenance.synthetic) $('#warn').innerHTML = '<div class="warn">⚠ Synthetic demonstration data — shows the workflow, not a real place.</div>';
else if ((S.provenance.socio||'').includes('PROXY')) $('#warn').innerHTML = '<div class="warn">⚠ No socio-economic data supplied — equity results are not meaningful.</div>';
const eqRow = k => S.equity.find(e => e.indicator === k);
function kpis(){ $('#kpis').innerHTML = ['overall', ...SC].map(k => { const v = S.scores[k], e = eqRow(k);
  return `<div class="kpi ${k===cur?'on':''}" data-k="${k}"><div class="l">${LBL[k]}</div><div class="v">${v.toFixed(0)}</div><div class="bar"><i style="width:${v}%;background:${scol(v)}"></i></div>
  <small>${e ? `deprived ${e.most_deprived_q.toFixed(0)} · affluent ${e.least_deprived_q.toFixed(0)}` : ''}</small></div>`; }).join('');
  document.querySelectorAll('.kpi').forEach(el => el.onclick = () => { setInd(el.dataset.k); show('map'); }); }
const TABS = [['map','Map'],['eq','Equity'],['zones','Zones & actions'],['j','Junctions'],['ty','Typologies & brief']];
$('#tabs').innerHTML = TABS.map(([k,l]) => `<button data-t="${k}" class="${k==='map'?'on':''}">${l}</button>`).join('');
function show(t){ document.querySelectorAll('section').forEach(s => s.classList.toggle('on', s.id === 't-' + t));
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  if (t === 'map') setTimeout(() => map.invalidateSize(), 50); if (t === 'eq') drawEq(); if (t === 'ty') drawTy(); }
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => show(b.dataset.t));

// ---------- map ----------
const map = L.map('map', {preferCanvas: true, zoomControl: true});
const base = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {attribution: '© OpenStreetMap © CARTO', maxZoom: 19}).addTo(map);
$('#oBase').onchange = e => e.target.checked ? base.addTo(map) : map.removeLayer(base);
const cellLayer = L.geoJSON(D.cells, {style: f => cellStyle(f), onEachFeature: (f, l) => { l.on('click', () => selectCell(f, l));
  l.bindTooltip(() => `${LBL[cur]}: ${f.properties[cur] == null ? 'n/a' : f.properties[cur].toFixed(1)}`, {sticky: true}); }}).addTo(map);
map.fitBounds(cellLayer.getBounds());
const zoneLayer = L.geoJSON(D.zones, {style: {color: dark ? '#fff' : '#111', weight: 2.5, dashArray: '6 4', fill: false, interactive: true},
  onEachFeature: (f, l) => l.bindTooltip(`${f.properties.name} · ${Math.round(f.properties.pop).toLocaleString()} residents`)}).addTo(map);
$('#oZones').onchange = e => e.target.checked ? zoneLayer.addTo(map) : map.removeLayer(zoneLayer);
const juncLayer = L.layerGroup();
function drawJunc(){ juncLayer.clearLayers(); const th = +$('#rs').value; let n = 0;
  D.junctions.forEach(j => { if (j.risk >= th) { n++; L.circleMarker([j.lat, j.lon], {radius: 3 + j.risk/25, color: ramp(['#1a9850','#fee08b','#d73027'], j.risk/80), weight: 1, fillOpacity: .85})
    .bindTooltip(`Junction #${j.node}: risk ${j.risk.toFixed(0)} · ${j.vmax_kmh.toFixed(0)} km/h · ${j.lanes} lanes · crossing: ${j.crossing}`).addTo(juncLayer); } });
  $('#rv').textContent = th; $('#rc').textContent = n; }
$('#oJunc').onchange = e => { if (e.target.checked) juncLayer.addTo(map); else map.removeLayer(juncLayer); };
function cellStyle(f){ const sel = selected && selected.properties.cell_id === f.properties.cell_id;
  return {fillColor: col(cur, f.properties[cur]), fillOpacity: .72, color: sel ? '#000' : '#fff', weight: sel ? 2.5 : .4}; }
function setInd(k){ cur = k; cellLayer.setStyle(cellStyle); kpis(); chips();
  const g = SEQ.has(k) ? SQ : DIV; $('#legend').style.background = `linear-gradient(90deg,${g.join(',')})`;
  $('#lg0').textContent = SEQ.has(k) ? 'low' : '0 (poor)'; $('#lg1').textContent = SEQ.has(k) ? 'high' : '100 (good)'; }
function chips(){ $('#ind').innerHTML = Object.keys(LBL).map(k => `<button data-k="${k}" class="${k===cur?'on':''}">${LBL[k]}</button>`).join('');
  document.querySelectorAll('#ind button').forEach(b => b.onclick = () => setInd(b.dataset.k)); }
function selectCell(f){ selected = f; cellLayer.setStyle(cellStyle); const p = f.properties, fx = (v, d = 0) => v == null ? '–' : (+v).toFixed(d);
  $('#cellPanel').innerHTML = `<h3>Cell ${p.cell_id} <span class="tag">${esc(p.typology)}</span></h3>` +
   ['overall', ...SC].map(k => `<div class="row"><span>${LBL[k]}</span><b>${fx(p[k])}</b></div><div class="bar"><i style="width:${p[k]||0}%;background:${scol(p[k]||0)}"></i></div>`).join('') +
   `<div style="margin-top:8px"><div class="row"><span>Population</span><b>${fx(p.pop)}</b></div><div class="row"><span>Deprivation (0–1)</span><b>${fx(p.deprivation,2)}</b></div>
    <div class="row"><span>Network distance to park</span><b>${fx(p.park_dist_m)} m</b></div><div class="row"><span>Surface temp (LST)</span><b>${fx(p.lst_c,1)} °C</b></div>
    <div class="row"><span>NDVI</span><b>${fx(p.ndvi,2)}</b></div><div class="row"><span>Pedshed ratio</span><b>${fx(p.pedshed,2)}</b></div>
    <div class="row"><span>Destination types in 10 min</span><b>${fx(p.dest_cats,0)} / 6</b></div><div class="row"><span>Transit accessibility index</span><b>${fx(p.ptal_ai,1)}</b></div>
    <div class="row"><span>Sidewalk coverage</span><b>${p.sidewalk_share == null ? '–' : Math.round(p.sidewalk_share*100) + '%'}</b></div>${p.is_priority ? '<p><span class="badge" style="background:#b2182b">Priority cell</span></p>' : ''}</div>`; }

// ---------- findings ----------
(function(){ const sc = S.scores, ks = SC.slice().sort((a,b) => sc[a]-sc[b]);
  const gaps = S.equity.filter(e => e.indicator !== 'overall').sort((a,b) => b.gap - a.gap);
  const L = [`Weakest dimension: <b>${LBL[ks[0]]}</b> (${sc[ks[0]].toFixed(0)}); strongest: <b>${LBL[ks[5]]}</b> (${sc[ks[5]].toFixed(0)}).`,
   `<b>${S.pct_pop_park300.toFixed(0)}%</b> of residents live within 300 m (on foot) of public green space.`,
   `<b>${S.junctions_high_risk}</b> of ${S.junctions} junctions screen as high pedestrian risk; <b>${S.junctions_unprotected_fast}</b> are on ≥40 km/h roads with no crossing.`];
  if (!(S.provenance.socio||'').includes('PROXY')) L.push(`Largest equity gap: <b>${LBL[gaps[0].indicator]}</b> — most-deprived quintile scores ${gaps[0].gap.toFixed(0)} pts lower.`);
  if (D.zones.features.length) L.push(`<b>${D.zones.features.length}</b> priority zones identified; top zone has ${Math.round(D.zones.features[0].properties.pop).toLocaleString()} residents.`);
  $('#ins').innerHTML = L.map(x => `<li>${x}</li>`).join(''); })();

// ---------- equity ----------
const order = F.map((f,i) => i).sort((a,b) => F[a].properties.deprivation - F[b].properties.deprivation);
order.forEach((i,r) => F[i].properties.q = Math.min(4, Math.floor(r*5/F.length)));
function qmean(k){ const s = [0,0,0,0,0], w = [0,0,0,0,0]; F.forEach(f => { const p = f.properties, v = p[k]; if (v == null || !(p.pop > 0)) return; s[p.q] += v*p.pop; w[p.q] += p.pop; }); return s.map((x,i) => w[i] ? x/w[i] : null); }
let cEq, cRad, cTy; const tc = () => getComputedStyle(document.body).color;
$('#eqInd').innerHTML = ['overall', ...SC].map(k => `<button data-k="${k}" class="${k===eqKey?'on':''}">${LBL[k]}</button>`).join('');
document.querySelectorAll('#eqInd button').forEach(b => b.onclick = () => { eqKey = b.dataset.k; document.querySelectorAll('#eqInd button').forEach(x => x.classList.toggle('on', x === b)); drawEq(); });
function drawEq(){ const q = qmean(eqKey); $('#eqT').textContent = LBL[eqKey] + ' by deprivation quintile';
  const opts = {responsive: true, maintainAspectRatio: false, plugins: {legend: {labels: {color: tc()}}}};
  if (cEq) cEq.destroy(); cEq = new Chart($('#cEq'), {type: 'bar', data: {labels: ['Q1 least','Q2','Q3','Q4','Q5 most deprived'], datasets: [{label: LBL[eqKey], data: q, backgroundColor: q.map(v => v == null ? '#999' : scol(v))}]},
    options: {...opts, plugins: {legend: {display: false}}, scales: {y: {min: 0, max: 100, ticks: {color: tc()}}, x: {ticks: {color: tc()}}}}});
  const qs = SC.map(k => qmean(k)); if (cRad) cRad.destroy();
  cRad = new Chart($('#cRad'), {type: 'radar', data: {labels: SC.map(k => LBL[k]), datasets: [{label: 'Least deprived (Q1)', data: qs.map(a => a[0]), borderColor: '#1b7837', backgroundColor: 'rgba(27,120,55,.15)'}, {label: 'Most deprived (Q5)', data: qs.map(a => a[4]), borderColor: '#b2182b', backgroundColor: 'rgba(178,24,43,.15)'}]},
    options: {...opts, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {color: tc(), font: {size: 10}}, grid: {color: 'rgba(128,128,128,.3)'}}}}}); }
$('#eqTab').innerHTML = '<tr><th>Indicator</th><th>Least-deprived</th><th>Most-deprived</th><th>Gap</th><th>Conc. index</th></tr>' +
  S.equity.map(e => `<tr><td>${LBL[e.indicator]}</td><td>${e.least_deprived_q.toFixed(1)}</td><td>${e.most_deprived_q.toFixed(1)}</td><td style="color:${e.gap > 5 ? '#b2182b' : 'inherit'}"><b>${e.gap > 0 ? '+' : ''}${e.gap.toFixed(1)}</b></td><td>${e.concentration_index > 0 ? '+' : ''}${e.concentration_index.toFixed(3)}</td></tr>`).join('');

// ---------- zones ----------
$('#zones').innerHTML = D.zones.features.length ? D.zones.features.map(f => { const z = f.properties, rs = D.recs.filter(r => r.zone === z.name);
  return `<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><div><b style="font-size:16px">${esc(z.name)}</b>
   <div class="mut">${Math.round(z.pop).toLocaleString()} residents · ${z.cells} cells · deprivation ${z.deprivation.toFixed(2)}</div></div>
   <button class="btn" data-z="${esc(z.name)}">Show on map</button></div>
   <div style="margin:8px 0">${SC.map(k => `<span class="badge" style="background:${scol(z[k])};margin:0 3px 3px 0;color:${z[k] > 35 && z[k] < 75 ? '#000' : '#fff'}">${LBL[k].split(' ')[0]} ${z[k].toFixed(0)}</span>`).join('')}</div>
   ${rs.map(r => `<details ${r.rank === 1 ? 'open' : ''}><summary>${r.rank}. ${esc(r.intervention)} <span class="tag">${esc(r.cost)} · ${esc(r.horizon)} · ${esc(r.confidence)} conf.</span></summary>
   <p>${esc(r.action)}</p><p><small><i>Evidence:</i> ${esc(r.evidence)}</small></p><p><small>Targets: ${esc(r.indicators)} · severity ${r.severity}</small></p></details>`).join('')}</div>`; }).join('')
  : '<div class="panel">No priority zones identified.</div>';
document.querySelectorAll('[data-z]').forEach(b => b.onclick = () => { const f = D.zones.features.find(x => x.properties.name === b.dataset.z);
  show('map'); setTimeout(() => map.fitBounds(L.geoJSON(f).getBounds().pad(.3)), 80); });

// ---------- junctions ----------
$('#rs').oninput = drawJunc; drawJunc();
$('#jt').innerHTML = '<tr><th>ID</th><th>Risk</th><th>km/h</th><th>Lanes</th><th>Crossing</th><th>Demand</th><th></th></tr>' +
  D.junctions.slice(0, 15).map((j, i) => `<tr><td>#${j.node}</td><td><span class="badge" style="background:${ramp(['#1a9850','#e0a800','#d73027'], j.risk/80)}">${j.risk.toFixed(0)}</span></td><td>${j.vmax_kmh.toFixed(0)}</td><td>${j.lanes}</td><td>${j.crossing}</td><td>${(j.demand*100).toFixed(0)}%</td><td><button class="btn" data-j="${i}">Map</button></td></tr>`).join('');
document.querySelectorAll('[data-j]').forEach(b => b.onclick = () => { const j = D.junctions[+b.dataset.j]; $('#oJunc').checked = true; juncLayer.addTo(map); show('map'); setTimeout(() => map.setView([j.lat, j.lon], 17), 80); });

// ---------- typologies & brief ----------
const TY = {}; F.forEach(f => { const p = f.properties; const t = TY[p.typology] = TY[p.typology] || {n: 0, pop: 0, o: 0}; t.n++; t.pop += p.pop || 0; t.o += (p.overall || 0) * (p.pop || 1); t.w = (t.w || 0) + (p.pop || 1); });
const tyArr = Object.entries(TY).sort((a,b) => a[1].o/a[1].w - b[1].o/b[1].w);
$('#tyTab').innerHTML = '<tr><th>Typology</th><th>Cells</th><th>Pop.</th><th>Score</th></tr>' + tyArr.map(([k, t]) => `<tr><td>${esc(k)}</td><td>${t.n}</td><td>${Math.round(t.pop).toLocaleString()}</td><td>${(t.o/t.w).toFixed(0)}</td></tr>`).join('');
function drawTy(){ if (cTy) cTy.destroy(); cTy = new Chart($('#cTy'), {type: 'doughnut', data: {labels: tyArr.map(a => a[0]), datasets: [{data: tyArr.map(a => Math.round(a[1].pop)), backgroundColor: ['#4e79a7','#f28e2b','#59a14f','#e15759','#b07aa1','#76b7b2','#edc948']}]},
  options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'bottom', labels: {color: tc(), boxWidth: 12, font: {size: 10}}}}}}); }
$('#brief').textContent = S.brief || ''; $('#bsrc').textContent = S.brief_source ? '(' + S.brief_source + ')' : '';

kpis(); chips(); setInd('overall');
</script></body></html>
"""

if __name__ == "__main__":
    build_dashboard(sys.argv[1] if len(sys.argv) > 1 else "out_demo")
