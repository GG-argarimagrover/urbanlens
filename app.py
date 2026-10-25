"""UrbanLens - Streamlit app. Type any city, get walkability / heat / junction safety /
transit / green-space / equity analysis, live. Deploy: push this repo to GitHub -> share.streamlit.io."""
import math
import os
import tempfile

import geopandas as gpd
import pandas as pd
import streamlit as st

import io

import numpy as np

import audit as au
import segments as sg
import streets_view as sv
import theme
import urbanlens as u

st.set_page_config(page_title="UrbanLens · Living Ganga", page_icon="🏔️", layout="wide")
st.markdown(theme.css(), unsafe_allow_html=True)

MAX_CELLS = 3000          # keeps free-tier hosting responsive
MAX_AREA_KM2 = 600        # refuse huge boundaries unless a bbox is given

# ------------------------------------------------------------------ helpers
def embed_html(html, height):
    if hasattr(st, "iframe"):                 # Streamlit >= 1.5x
        st.iframe(html, height=height)
    else:                                      # older versions
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)


def save_upload(f, tmp):
    if f is None:
        return None
    p = os.path.join(tmp, f.name)
    with open(p, "wb") as out:
        out.write(f.getbuffer())
    return p


def check_socio(path):
    p = ("zip://" + path) if path.lower().endswith(".zip") else path
    g = gpd.read_file(p)
    missing = [c for c in ("population", "deprivation") if c not in g.columns]
    if missing:
        raise ValueError(f"Ward file is missing column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(c for c in g.columns if c != 'geometry')}")
    if g["deprivation"].max() > 1.0001 or g["deprivation"].min() < -0.0001:
        raise ValueError("'deprivation' must be between 0 and 1 (higher = worse off). Rescale it first.")
    return p


@st.cache_resource(show_spinner=False, max_entries=3)
def load_city_cached(place, bbox, socio, pop, ndvi, lst):
    return u.load_osm(place, socio, ndvi, lst, pop, bbox or None)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### 🏔️ UrbanLens")
st.sidebar.caption("Living Ganga · urban intelligence for any city")
mode = st.sidebar.radio("Data source", ["Any city (OpenStreetMap)", "Demo (synthetic city)"],
                        help="The demo needs no internet and shows how the app works.")
place = bbox = ""
if mode.startswith("Any"):
    place = st.sidebar.text_input("City / town", "Rishikesh, Uttarakhand, India",
                                  help="Use 'Town, State, Country'. Keep to a town or district-sized area.")
    bbox = st.sidebar.text_input("…or bounding box (optional)", "", placeholder="west,south,east,north",
                                 help="Use this if the name lookup picks the wrong area or is too large.")
cell = st.sidebar.select_slider("Grid cell size (m)", [150, 200, 250, 300, 400, 500], value=250,
                                help="Smaller = more detail but slower. Auto-increased for big areas.")

seg_len = st.sidebar.select_slider("Street segment length (m)", [25, 50, 75, 100], value=50,
                                   help="Streets are cut into segments of this length for the Happy Streets scoring. Auto-increased for very large networks.")

with st.sidebar.expander("Optional: your own data (improves accuracy)"):
    f_socio = st.file_uploader("Ward / zone polygons (.gpkg, .geojson, zipped shapefile)", type=["gpkg", "geojson", "json", "zip"],
                               help="Needs columns: population, deprivation (0-1, higher = worse). Optional: elderly_share (0-1). "
                                    "Without this, the Equity analysis is disabled.")
    f_pop = st.file_uploader("Population raster (GeoTIFF, e.g. WorldPop)", type=["tif", "tiff"])
    f_ndvi = st.file_uploader("NDVI raster (GeoTIFF)", type=["tif", "tiff"])
    f_lst = st.file_uploader("Land-surface temperature raster (GeoTIFF)", type=["tif", "tiff"])

with st.sidebar.expander("Advanced: indicator weights"):
    W = {k: st.slider(u.SCORE_LABELS[k].split(" &")[0], 0, 10, int(u.DEFAULT_WEIGHTS[k] * 100 / 5)) for k in u.SCORES}
    if sum(W.values()) == 0:
        W = {k: 1 for k in u.SCORES}

def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


use_llm = False
if os.environ.get("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API_KEY"):
    use_llm = st.sidebar.checkbox("AI-written planning brief (Claude)", value=False)
    if use_llm and _secret("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = _secret("ANTHROPIC_API_KEY")

run_clicked = st.sidebar.button("▶ Analyse", type="primary", width="stretch")

# ------------------------------------------------------------------ run
if run_clicked:
    st.session_state.pop("result", None)
    tmp = tempfile.mkdtemp()
    try:
        with st.status("Running analysis…", expanded=True) as status:
            eff_cell = cell
            if mode.startswith("Demo"):
                st.write("Generating synthetic city…")
                city = u.synthetic_city(os.path.join(tmp, "rs"))
            else:
                if not place.strip() and not bbox.strip():
                    raise ValueError("Enter a city name or a bounding box.")
                socio = pop = ndvi = lst = None
                if f_socio is not None:
                    socio = check_socio(save_upload(f_socio, tmp))
                pop, ndvi, lst = (save_upload(f, tmp) for f in (f_pop, f_ndvi, f_lst))
                if bbox.strip():
                    w, s, e, n = (float(v) for v in bbox.split(","))
                    if not (w < e and s < n):
                        raise ValueError("Bounding box must be west,south,east,north with west<east and south<north.")
                    area = (e - w) * 111.0 * math.cos(math.radians((s + n) / 2)) * (n - s) * 111.0
                else:
                    st.write(f"Looking up “{place}”…")
                    area, _ = u.place_area_km2(place)
                st.write(f"Area ≈ {area:,.0f} km²")
                if area > MAX_AREA_KM2:
                    raise ValueError(f"That area is {area:,.0f} km² - too large for this app (max {MAX_AREA_KM2}). "
                                     "Enter a bounding box around the built-up area instead.")
                need = math.ceil(math.sqrt(area * 1e6 / MAX_CELLS) / 50) * 50
                if need > cell:
                    eff_cell = need
                    st.write(f"Large area → cell size raised to {eff_cell} m to stay responsive.")
                st.write("Downloading OpenStreetMap data (streets, shops, stops, parks, buildings)… "
                         "this is the slow step - a few minutes for a town.")
                city = load_city_cached(place.strip(), bbox.strip(), socio, pop, ndvi, lst)
            st.write("Computing network accessibility, safety, heat, equity, recommendations…")
            res = u.run(city, tmp, cell=eff_cell, use_llm=use_llm, weights=W)
            st.write("Scoring street segments (Happy Streets Index)…")
            seg = sg.build_segments(city, res.cells, res.junctions, seg_len=seg_len)
            status.update(label="Done", state="complete", expanded=False)
        rd = lambda n, m="r": open(os.path.join(tmp, n), m, **({"encoding": "utf-8"} if m == "r" else {})).read()
        st.session_state.result = dict(
            summary=res.summary, dashboard=rd("dashboard.html"), report=rd("report.html"),
            recs=res.recs, equity=res.equity, coverage=u.coverage_report(city), cell=eff_cell,
            cells_geojson=rd("cells.geojson", "rb"), junctions_geojson=rd("junctions.geojson", "rb"),
            recs_csv=rd("recommendations.csv", "rb"), name=city.name, seg=seg)
    except Exception as e:
        st.error(f"**Could not complete the analysis.** {e}")
        if mode.startswith("Any"):
            st.info("Tips: check spelling (‘Town, State, Country’), try a bounding box, or try again - "
                    "the OpenStreetMap servers are sometimes busy.")

# ------------------------------------------------------------------ results
R = st.session_state.get("result")
if not R:
    st.markdown(theme.banner_html("Understand any city, street by street"), unsafe_allow_html=True)
    st.markdown("""
Type a city or town in the sidebar and press **Analyse**. The app downloads open map data and measures:

- 🚶 **Walkability & street accessibility** (network-based, not straight-line)
- 🌡️ **Urban heat & climate vulnerability**
- 🚦 **Junction safety & pedestrian comfort**
- 🚌 **Public transport connectivity**
- 🌳 **Green-space access & neighbourhood equity**

…then finds priority zones and suggests evidence-linked actions. New here? Try the **Demo** first.
""")
    st.markdown(theme.footer_html(), unsafe_allow_html=True)
    st.stop()

S = R["summary"]
st.markdown(theme.banner_html(S["city"], "Urban intelligence report · scores 0-100, higher is better", compact=True), unsafe_allow_html=True)
prov = S["provenance"].get("socio", "")
if S["provenance"].get("synthetic"):
    st.warning("Synthetic demonstration data - shows how the app works, not a real place.")
elif "PROXY" in prov:
    st.warning("No deprivation data supplied → **Equity results are not meaningful.** Upload ward data in the sidebar to enable them.")
low = R["coverage"][R["coverage"]["Status"] == "LOW"]
if len(low) and not S["provenance"].get("synthetic"):
    st.info(f"⚠ Thin OpenStreetMap coverage for: {', '.join(low['Layer'])}. See the *Data quality* tab before trusting those scores.")

cols = st.columns(7)
cols[0].metric("Overall", f"{S['scores']['overall']:.0f}")
for c_, k in zip(cols[1:], u.SCORES):
    c_.metric(u.SCORE_LABELS[k].split(" &")[0], f"{S['scores'][k]:.0f}")

t1, t1b, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "🛣️ Happy Streets", "✅ Recommendations", "🔎 Data quality", "⬇ Downloads", "ℹ Method"])
with t1:
    st.caption("Tap a KPI card to change the map layer; tap cells for details; use the tabs inside for equity, zones and junctions.")
    embed_html(R["dashboard"], 1150)

with t1b:
    seg0 = R.get("seg")
    if seg0 is None or not len(seg0):
        st.info("Run the analysis again to generate street segments.")
    else:
        st.markdown("**Street-by-street scoring.** Every street is cut into short segments and scored on pedestrian infrastructure, "
                    "traffic exposure, shade, lighting, vitality and stayability. Each score shows its **evidence basis** - "
                    "measured (field audit / OSM tag / satellite), proxy, or assumed. Perceived safety and accessibility are only scored "
                    "when you upload field audits.")
        # ---- weights
        with st.expander("Indicator weights"):
            Wsg = {k: st.slider(sg.LABELS[k].replace(" (audit only)", ""), 0, 10, int(sg.DEFAULT_WEIGHTS[k] * 100 / 5), key=f"w_{k}")
                   for k in sg.INDICATORS}
            if sum(Wsg.values()) == 0:
                Wsg = {k: 1 for k in sg.INDICATORS}
        # ---- audit upload
        agg = None
        up = st.file_uploader("Upload completed field audits (Excel template, CSV, or KoboToolbox export)", type=["xlsx", "csv"], key="audit_up")
        if up is not None:
            try:
                adf, awarn = au.read_audit(up)
                agg, arep = au.aggregate_audit(adf, seg0)
                st.success(f"{arep['segments_audited']} segments audited ({arep['matched_by_id']} matched by ID, "
                           f"{arep['matched_by_location']} by GPS; {arep['unmatched']} unmatched).")
                for w_ in awarn:
                    st.warning(w_)
            except Exception as e:
                st.error(f"Could not read the audit file: {e}")
        SG = sg.score_segments(seg0, weights=Wsg, audit=agg)
        C = sg.corridors(SG)
        L = SG.length_m
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Happy Streets Index", f"{(SG.hsi * L).sum() / L.sum():.0f}", help="Length-weighted mean, 0-100")
        k2.metric("Street length scoring poor (<40)", f"{100 * (L * (SG.hsi < 40)).sum() / L.sum():.0f}%")
        k3.metric("Segments field-audited", f"{100 * SG.audited.mean():.0f}%")
        k4.metric("Medium/high-confidence length", f"{100 * (L * SG.confidence.isin(['Medium', 'High'])).sum() / L.sum():.0f}%",
                  help="Confidence is High if ≥50% of the score is audited, Medium if ≥50% is measured (audit/OSM tag/satellite), otherwise Low.")

        # ---- scenario simulator
        show_scn = False
        with st.expander("🔧 Before / after intervention simulator (MODELLED, not measured)"):
            st.caption("Applies assumed effects to the selected corridors and recomputes the index. These are scenario estimates, "
                       "not approved plans or measured outcomes. Edit the assumptions to match local evidence.")
            sel_c = st.multiselect("Corridors to treat", list(C.corridor), default=list(C.corridor.head(5)))
            acts = st.multiselect("Interventions", list(sg.SCENARIO_ACTIONS), default=["footpath", "trees"],
                                  format_func=lambda a: sg.SCENARIO_ACTIONS[a])
            c1_, c2_, c3_ = st.columns(3)
            eff = dict(footpath_target=c1_.slider("Footpath score after works", 50, 100, 80),
                       calm_speed=c1_.slider("Target speed (km/h)", 10, 50, 30),
                       trees_gain=c2_.slider("Shade score gain", 5, 60, 30),
                       light_target=c2_.slider("Lighting score after works", 50, 100, 80),
                       seating_gain=c3_.slider("Stayability gain", 5, 50, 25),
                       activate_gain=c3_.slider("Vitality gain", 5, 50, 20))
            st.caption("Optional unit costs per km of street treated (your currency; leave 0 to skip costing):")
            ucols = st.columns(3)
            ucost = {a: ucols[i % 3].number_input(sg.SCENARIO_ACTIONS[a], min_value=0.0, value=0.0, step=1000.0, key=f"uc_{a}")
                     for i, a in enumerate(sg.SCENARIO_ACTIONS)}
            if sel_c and acts:
                T, sr = sg.simulate(SG, sel_c, acts, eff, Wsg, ucost)
                m1, m2, m3 = st.columns(3)
                m1.metric("Index: before → after", f"{sr['hsi_before']:.1f} → {sr['hsi_after']:.1f}", f"{sr['delta']:+.1f}")
                m2.metric("Poor street length", f"{sr['poor_before']:.0f}% → {sr['poor_after']:.0f}%")
                m3.metric("Indicative cost", "enter unit costs" if sr["cost"] is None else f"{sr['cost']:,.0f}")
                st.dataframe(pd.DataFrame({"Intervention": [sg.SCENARIO_ACTIONS[a] for a in acts],
                                           "Street length treated (km)": [round(sr["km_treated"][a], 1) for a in acts]}),
                             hide_index=True, width="stretch")
                show_scn = st.checkbox("Show the scenario on the map")

        # ---- map
        note = ("Synthetic demonstration data. " if R["summary"]["provenance"].get("synthetic") else "") + \
               ("MODELLED SCENARIO - not measured." if show_scn else "")
        embed_html(sv.build_streets_html(T if show_scn else SG, C, note=note.strip()), 860)

        # ---- corridors & recommendations
        st.subheader("Priority corridors")
        show = C.head(25)[["corridor", "length_m", "hsi", "demand", "priority_score", "audited_pct", "confidence_high_pct"]]
        st.dataframe(show.rename(columns={"corridor": "Corridor", "length_m": "Length (m)", "hsi": "HSI", "demand": "Ped. demand",
                                          "priority_score": "Priority score", "audited_pct": "% audited",
                                          "confidence_high_pct": "% high confidence"}), hide_index=True, width="stretch")
        RC = sg.corridor_recommendations(C)
        for cname, g in RC.groupby("corridor", sort=False):
            r0 = g.iloc[0]
            with st.expander(f"**#{int(r0.corridor_rank)} {cname}** - HSI {r0.corridor_hsi:.0f}, {r0.length_m / 1000:.1f} km"):
                st.caption(f"Evidence: {r0.evidence_confidence}")
                for _, r in g.iterrows():
                    st.markdown(f"**{r['rank']}. {r.intervention}** - on {r.share_of_length_pct}% of the corridor  "
                                f"`{r.cost_band} · {r.horizon}`")
                    st.write(r.action)
                    st.caption(f"Typical lead: {r.typical_lead_agency} · Monitor: {r.monitoring_indicator}")

        # ---- field audit kit + exports
        st.subheader("Field audit kit & exports")
        n_aud = st.slider("Segments to include in the audit sheet", 20, 300, 100, 10,
                          help="Chosen where an audit would change the picture most: busy, low-scoring, not yet audited.")
        buf = io.BytesIO(); au.make_template_xlsx(SG, buf, n=n_aud)
        buf2 = io.BytesIO(); au.make_xlsform(buf2)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Audit sheet (Excel, pre-filled)", buf.getvalue(), "happy_streets_audit_sheet.xlsx")
        d2.download_button("KoboToolbox form (XLSForm)", buf2.getvalue(), "happy_streets_audit_form.xlsx")
        d3.download_button("Corridors (CSV)", C.to_csv(index=False).encode(), "corridors.csv", "text/csv")
        d4.download_button("Segments (GeoJSON)", SG.to_crs(4326).to_json().encode(), "segments.geojson", "application/geo+json")
        st.caption("Segment IDs are tied to this data snapshot: keep the GeoJSON/Excel you audit against. "
                   "Audit rows that carry only GPS coordinates are matched to the nearest segment (within 40 m).")

with t2:
    st.markdown(S.get("brief", ""))
    if len(R["recs"]):
        for zone, g in R["recs"].groupby("zone", sort=True):
            with st.expander(f"**{zone}** - {int(g['population'].iloc[0]):,} residents", expanded=(zone == R["recs"]["zone"].iloc[0])):
                for _, r in g.iterrows():
                    st.markdown(f"**{int(r['rank'])}. {r['intervention']}**  `{r['cost']} · {r['horizon']} · {r['confidence']} confidence`")
                    st.write(r["action"])
                    st.caption(f"Evidence: {r['evidence']}")
    else:
        st.write("No priority zones found.")
with t3:
    st.dataframe(R["coverage"], hide_index=True, width="stretch")
    st.caption("OpenStreetMap completeness varies by city. 'LOW' means the related score may be biased - "
               "supply your own data in the sidebar or treat results as indicative.")
    st.json(S["provenance"])
with t4:
    st.download_button("Dashboard (HTML, shareable)", R["dashboard"], "dashboard.html", "text/html")
    st.download_button("Report (HTML, printable)", R["report"], "report.html", "text/html")
    st.download_button("Recommendations (CSV)", R["recs_csv"], "recommendations.csv", "text/csv")
    st.download_button("Cells with all indicators (GeoJSON)", R["cells_geojson"], "cells.geojson", "application/geo+json")
    st.download_button("Junctions with risk (GeoJSON)", R["junctions_geojson"], "junctions.geojson", "application/geo+json")
with t5:
    st.markdown(f"""
**Grid:** {R['cell']} m cells · **Scores:** 0-100, higher = better.
- **Walkability:** intersection density, network pedshed (800 m), destination types within a 10-minute walk.
- **Junction safety:** screening heuristic - pedestrian injury-severity vs speed × road width × legs × crossing/sidewalk mitigation. *Not a crash model.*
- **Comfort:** footway coverage, low-speed streets, arterial exposure, shade.
- **Transit:** PTAL-inspired index. OSM has no timetables → frequencies are **placeholders** unless you supply GTFS.
- **Green:** network distance to ≥0.5 ha green space (3-30-300 rule), vegetation, park area per person.
- **Heat:** exposure (LST/vegetation/paving) + sensitivity − adaptive capacity.
- **Equity:** quintile gaps & concentration index - requires your deprivation data.
- **Happy Streets Index (street segments):** 25-100 m segments scored on infrastructure, traffic exposure, shade, lighting, vitality, stayability (open data) plus perceived safety and accessibility (**field audit only**). Each score is tagged audit / OSM / satellite / proxy / default.
- **AI layer:** k-means neighbourhood typologies + rule-based recommendation engine (every action cites the metrics behind it).

Hills/slope are not modelled. Treat results as screening, then verify on the ground.""")

st.markdown(theme.footer_html(), unsafe_allow_html=True)
