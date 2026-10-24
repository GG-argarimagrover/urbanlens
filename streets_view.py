"""Happy Streets Explorer - self-contained interactive HTML for scored street segments."""
import json

import numpy as np
import geopandas as gpd

import segments as sg


def build_streets_html(S, C, title="Happy Streets Explorer", note=""):
    S4 = S.to_crs(4326)
    segs, coords = [], []
    r1 = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 1)
    for _, r in S4.iterrows():
        segs.append(dict(
            id=r.seg_id, n=r["name"] or "(unnamed)", c=r.corridor, hw=r.highway, len=round(r.length_m), hsi=r1(r.hsi),
            pri=r1(r.priority), dem=round(float(r.demand), 2), conf=r.confidence, aud=int(r.audited), iv=r.iv,
            spd=round(r.speed), s=[r1(r[f"s_{k}"]) for k in sg.INDICATORS],
            b=["".join(x[0] for x in [r[f"b_{k}"]] if x) for k in sg.INDICATORS],
            sp=r1(r.safety_proxy)))
        coords.append([[round(x, 5), round(y, 5)] for x, y in r.geometry.coords])
    corr = [dict(c=r.corridor, hsi=r.hsi, ps=r.priority_score, len=int(r.length_m)) for _, r in C.head(12).iterrows()]
    data = dict(segs=segs, coords=coords, corr=corr, title=title, note=note,
                labels=[sg.LABELS[k] for k in sg.INDICATORS], keys=sg.INDICATORS,
                iv={k: v["title"] for k, v in sg.INTERVENTIONS.items()})
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", blob)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--teal:#164E50;--forest:#527A62;--mist:#F3F5F2;--sand:#E8C68A;--saffron:#E87545;--bg:#F3F5F2;--panel:#fff;--fg:#17211f;--mut:#5d6b66;--line:#dfe4e0;--chip:#e6ebe7}
@media (prefers-color-scheme:dark){:root{--bg:#10201f;--panel:#182b2a;--fg:#e9efec;--mut:#9fb2ab;--line:#2b4543;--chip:#223a38}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 Inter,system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg)}
header{background:var(--teal);color:#fff;padding:10px 14px}header h1{margin:0;font-size:17px}header div{font-size:12px;opacity:.85}
.note{background:var(--sand);color:#3a2d10;padding:6px 12px;font-size:12px}
.wrap{display:grid;grid-template-columns:1fr;gap:10px;padding:10px}
@media(min-width:900px){.wrap{grid-template-columns:2fr 1fr}}
#map{height:62vh;min-height:360px;border-radius:12px;border:1px solid var(--line)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}.chips button{border:0;background:var(--chip);color:var(--fg);padding:5px 10px;border-radius:14px;font:inherit;font-size:12px;cursor:pointer}
.chips button.on{background:var(--teal);color:#fff}
.panel{background:var(--panel);border-radius:12px;padding:12px;margin-bottom:10px;border:1px solid var(--line)}
.panel h3{margin:0 0 6px;font-size:14px}.row{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px solid var(--line);font-size:13px}.row:last-child{border:0}
.bar{height:5px;background:var(--line);border-radius:3px;margin-bottom:2px}.bar i{display:block;height:5px;border-radius:3px}
.b{display:inline-block;font-size:10px;font-weight:700;padding:0 5px;border-radius:7px;color:#fff;margin-left:4px}
.ba{background:#164E50}.bo{background:#527A62}.br{background:#2f7f9d}.bp{background:#b07d1a}.bd{background:#9aa5a1}
.legend{height:10px;border-radius:5px;margin:8px 0 2px}.lg{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}
.corr{cursor:pointer;padding:5px 0;border-bottom:1px solid var(--line);font-size:13px}.corr:hover{color:var(--saffron)}
small,.mut{color:var(--mut)}label{font-size:12px}input[type=range]{width:100%}
.tag{display:inline-block;background:var(--chip);border-radius:9px;padding:1px 7px;font-size:11px;margin:2px 3px 2px 0}
</style></head><body>
<header><h1 id="t"></h1><div>Street-segment scores · tap a street for evidence and suggested actions</div></header>
<div id="note"></div>
<div class="wrap"><div>
 <div class="chips" id="chips"></div><div id="map"></div>
 <div class="legend" id="lgd"></div><div class="lg"><span id="l0"></span><span id="l1"></span></div>
 <div style="margin-top:8px"><label>Highlight segments scoring below <b id="thv">100</b> <input type="range" id="th" min="10" max="100" value="100"></label>
 <label><input type="checkbox" id="onlyAud"> only field-audited segments</label> · <label><input type="checkbox" id="base" checked> basemap</label></div>
</div><div>
 <div class="panel" id="sel"><h3>Tap a street segment</h3><div class="mut">Colour shows the selected indicator. The letter badges show the evidence behind each score.</div>
  <p><span class="b ba">A</span> field audit <span class="b bo">O</span> OSM tag <span class="b br">R</span> satellite <span class="b bp">P</span> proxy <span class="b bd">D</span> assumed default</p></div>
 <div class="panel"><h3>Top priority corridors</h3><div id="corr"></div><small>Priority = shortfall × pedestrian demand × length. Tap to zoom.</small></div>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D=__DATA__, $=s=>document.querySelector(s), esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('#t').textContent=D.title; if(D.note) $('#note').innerHTML='<div class="note">'+esc(D.note)+'</div>';
const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
const RAMP=['#b2182b','#E87545','#E8C68A','#a9c4a0','#527A62','#164E50'];
function ramp(st,t){t=Math.max(0,Math.min(1,t));const x=t*(st.length-1),i=Math.min(st.length-2,Math.floor(x)),a=hex(st[i]),b=hex(st[i+1]),f=x-i;return 'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')+')';}
const BASIS={a:'ba',o:'bo',r:'br',p:'bp',d:'bd'};
const LAYERS=[['hsi','Happy Streets Index'],...D.keys.map((k,i)=>[k,D.labels[i].replace(' (audit only)','')]),['pri','Priority'],['conf','Evidence confidence']];
let cur='hsi',selId=null;
const map=L.map('map',{preferCanvas:true});
const tiles=L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:'© OpenStreetMap © CARTO',maxZoom:19}).addTo(map);
$('#base').onchange=e=>e.target.checked?tiles.addTo(map):map.removeLayer(tiles);
const maxPri=Math.max(...D.segs.map(s=>s.pri||0))||1;
function val(s,k){return k==='hsi'?s.hsi:k==='pri'?s.pri:k==='conf'?null:s.s[D.keys.indexOf(k)];}
function colour(s){
  if(cur==='conf') return {High:'#164E50',Medium:'#E8C68A',Low:'#E87545'}[s.conf];
  const v=val(s,cur); if(v==null) return '#b8c0bc';
  return cur==='pri'?ramp(['#F3F5F2','#E8C68A','#E87545','#7a1f0e'],v/maxPri):ramp(RAMP,v/100);}
const lines=D.segs.map((s,i)=>{const l=L.polyline(D.coords[i].map(c=>[c[1],c[0]]),{weight:4,opacity:.9,color:'#999'}).addTo(map);
  l.on('click',()=>pick(i));l.bindTooltip(()=>s.n+' · '+(cur==='conf'?s.conf:(val(s,cur)==null?'n/a':val(s,cur))),{sticky:true});return l;});
function paint(){const th=+$('#th').value,oa=$('#onlyAud').checked;$('#thv').textContent=th;
  D.segs.forEach((s,i)=>{const v=cur==='conf'||cur==='pri'?s.hsi:val(s,cur);const hide=(oa&&!s.aud)||(v!=null&&v>th);
    lines[i].setStyle({color:colour(s),opacity:hide?.12:.92,weight:(selId===i)?7:(hide?2:4)});});}
function setLayer(k){cur=k;chips();paint();const g=k==='pri'?['#F3F5F2','#E8C68A','#E87545','#7a1f0e']:RAMP;
  $('#lgd').style.background=k==='conf'?'linear-gradient(90deg,#E87545,#E8C68A,#164E50)':'linear-gradient(90deg,'+g.join(',')+')';
  $('#l0').textContent=k==='conf'?'Low':k==='pri'?'low':'0 (poor)';$('#l1').textContent=k==='conf'?'High':k==='pri'?'high':'100 (good)';}
function chips(){$('#chips').innerHTML=LAYERS.map(([k,l])=>`<button data-k="${k}" class="${k===cur?'on':''}">${esc(l)}</button>`).join('');
  document.querySelectorAll('#chips button').forEach(b=>b.onclick=()=>setLayer(b.dataset.k));}
function pick(i){selId=i;const s=D.segs[i];paint();lines[i].bringToFront();
  const rows=D.keys.map((k,j)=>{const v=s.s[j],bs=s.b[j];return `<div class="row"><span>${esc(D.labels[j])}${bs?`<span class="b ${BASIS[bs]}">${bs.toUpperCase()}</span>`:''}</span><b>${v==null?'–':v.toFixed(0)}</b></div>${v==null?'':`<div class="bar"><i style="width:${v}%;background:${ramp(RAMP,v/100)}"></i></div>`}`}).join('');
  const iv=s.iv?s.iv.split(',').map(c=>`<span class="tag">${esc(D.iv[c]||c)}</span>`).join(''):'<span class="mut">No intervention flagged on current evidence</span>';
  $('#sel').innerHTML=`<h3>${esc(s.n)} <small>${s.id}</small></h3><div class="row"><span>Happy Streets Index</span><b style="font-size:18px">${s.hsi==null?'–':s.hsi.toFixed(0)}</b></div>
   <div class="mut">${s.len} m · ${esc(s.hw)} · ${s.spd} km/h · pedestrian demand ${(s.dem*100).toFixed(0)}% · evidence confidence <b>${s.conf}</b>${s.aud?' · field-audited':' · <b>not audited</b>'}</div><div style="margin-top:6px">${rows}</div>
   <p style="margin:8px 0 2px"><b>Suggested actions (screening)</b></p>${iv}
   <p class="mut" style="margin-top:6px">Indicative safety proxy (not scored, lighting+vitality+traffic): ${s.sp??'–'}. Perceived safety and accessibility need a field audit.</p>`;}
$('#corr').innerHTML=D.corr.map((c,i)=>`<div class="corr" data-c="${esc(c.c)}"><b>${i+1}. ${esc(c.c)}</b><br><small>HSI ${c.hsi.toFixed(0)} · ${(c.len/1000).toFixed(1)} km · priority ${c.ps}</small></div>`).join('');
document.querySelectorAll('.corr').forEach(el=>el.onclick=()=>{const idx=D.segs.map((s,i)=>s.c===el.dataset.c?i:-1).filter(i=>i>=0);
  if(idx.length){map.fitBounds(L.featureGroup(idx.map(i=>lines[i])).getBounds().pad(.15));pick(idx[0]);}});
$('#th').oninput=paint;$('#onlyAud').onchange=paint;
map.fitBounds(L.featureGroup(lines).getBounds());setLayer('hsi');
</script></body></html>
"""
