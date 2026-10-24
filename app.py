"""UrbanLens - Streamlit app. Type any city, get walkability / heat / junction safety /
transit / green-space / equity analysis, live. Deploy: push this repo to GitHub -> share.streamlit.io."""
import math
import os
import tempfile

import geopandas as gpd
import pandas as pd
import streamlit as st

import urbanlens as u

st.set_page_config(page_title="UrbanLens - city analytics", page_icon="🏙️", layout="wide")

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
st.sidebar.title("🏙️ UrbanLens")
st.sidebar.caption("Walkability · Heat · Junction safety · Transit · Green space · Equity")
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
            status.update(label="Done", state="complete", expanded=False)
        rd = lambda n, m="r": open(os.path.join(tmp, n), m, **({"encoding": "utf-8"} if m == "r" else {})).read()
        st.session_state.result = dict(
            summary=res.summary, dashboard=rd("dashboard.html"), report=rd("report.html"),
            recs=res.recs, equity=res.equity, coverage=u.coverage_report(city), cell=eff_cell,
            cells_geojson=rd("cells.geojson", "rb"), junctions_geojson=rd("junctions.geojson", "rb"),
            recs_csv=rd("recommendations.csv", "rb"), name=city.name)
    except Exception as e:
        st.error(f"**Could not complete the analysis.** {e}")
        if mode.startswith("Any"):
            st.info("Tips: check spelling (‘Town, State, Country’), try a bounding box, or try again - "
                    "the OpenStreetMap servers are sometimes busy.")

# ------------------------------------------------------------------ results
R = st.session_state.get("result")
if not R:
    st.title("UrbanLens - understand any city")
    st.markdown("""
Type a city or town in the sidebar and press **Analyse**. The app downloads open map data and measures:

- 🚶 **Walkability & street accessibility** (network-based, not straight-line)
- 🌡️ **Urban heat & climate vulnerability**
- 🚦 **Junction safety & pedestrian comfort**
- 🚌 **Public transport connectivity**
- 🌳 **Green-space access & neighbourhood equity**

…then finds priority zones and suggests evidence-linked actions. New here? Try the **Demo** first.
""")
    st.stop()

S = R["summary"]
st.title(S["city"])
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

t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "✅ Recommendations", "🔎 Data quality", "⬇ Downloads", "ℹ Method"])
with t1:
    st.caption("Tap a KPI card to change the map layer; tap cells for details; use the tabs inside for equity, zones and junctions.")
    embed_html(R["dashboard"], 1150)
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
- **AI layer:** k-means neighbourhood typologies + rule-based recommendation engine (every action cites the metrics behind it).

Hills/slope are not modelled. Treat results as screening, then verify on the ground.""")
