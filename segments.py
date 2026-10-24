"""Street-segment scoring - the Happy Streets Index (HSI).

Streets are cut into 25-100 m segments. Each segment gets up to eight indicator scores (0-100, higher = better),
and every score records its EVIDENCE BASIS so users can see what was measured and what was assumed:

    audit    field-audit observation supplied by the user
    osm      explicit OpenStreetMap tag (e.g. sidewalk=both, maxspeed=30, lit=yes)
    remote   satellite-derived raster (NDVI / land-surface temperature)
    proxy    indirect open-data proxy (e.g. shop doors as 'active frontage')
    default  an assumed value because nothing was tagged (adjustable parameter)

Perceived safety and universal accessibility cannot be measured from open data: they are only scored when a
field audit exists. Otherwise they are left out of the index (and the weights renormalised), never invented.
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

import urbanlens as u

INDICATORS = ["infra", "traffic", "shade", "lighting", "vitality", "stay", "safety_perc", "access"]
LABELS = {
    "infra": "Pedestrian infrastructure", "traffic": "Traffic exposure & crossing comfort",
    "shade": "Thermal comfort & shade", "lighting": "Lighting", "vitality": "Street vitality (active frontage)",
    "stay": "Stayability (seating, green)", "safety_perc": "Perceived safety (audit only)",
    "access": "Universal accessibility (audit only)",
}
DEFAULT_WEIGHTS = dict(infra=.20, traffic=.20, shade=.15, lighting=.10, vitality=.10, stay=.10, safety_perc=.10, access=.05)
BASES = ["audit", "osm", "remote", "proxy", "default"]
PED_HW = {"footway", "pedestrian", "path", "living_street", "cycleway"}
PAVED = {"asphalt", "paved", "concrete", "paving_stones", "sett", "concrete:plates", "concrete:lanes", "metal", "bricks"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "earth", "sand", "grass", "mud", "compacted", "fine_gravel", "pebblestone"}

PARAMS = dict(
    unknown_sidewalk_score=40,   # score assumed when sidewalk is not tagged at all (flagged 'default')
    unknown_lighting_score=50,   # same for lighting
    width_poor_m=1.0, width_good_m=1.8,   # clear footpath width thresholds - set to your local standard
    shade_target_pct=50, seating_per_100m_target=2.0, frontage_target_pct=70,
)

INTERVENTIONS = {
    "footpath": dict(title="Provide / complete continuous footpaths", cost="€€", horizon="1-3 yrs",
                     agency="ULB / PWD (roads)", monitor="% of segment length with continuous clear footpath",
                     action="Build or complete a continuous footpath of the locally required clear width, with kerb ramps at crossings; "
                            "remove obstructions and parking from the walking path."),
    "calm": dict(title="Traffic calming & safer crossings", cost="€€", horizon="0-2 yrs",
                 agency="ULB / PWD / traffic police", monitor="85th-percentile speed; pedestrian-vehicle conflicts",
                 action="Reduce speeds to ≤30 km/h (signage, raised crossings/tables, narrower carriageway, tighter corners) and add "
                        "protected crossings at the junctions on this street."),
    "trees": dict(title="Street trees & shade", cost="€€", horizon="2-5 yrs",
                  agency="ULB horticulture / forest dept", monitor="% of path shaded at 12:00-15:00; LST",
                  action="Plant trees on the walking route where space allows; use shade structures and cool/permeable surfaces where it does not."),
    "light": dict(title="Pedestrian-scale lighting", cost="€€", horizon="0-2 yrs",
                  agency="ULB electrical wing", monitor="Lux on footpath at night; % lamps working",
                  action="Install or repair pedestrian-scale lighting, prioritising crossings, stops and dark stretches."),
    "activate": dict(title="Activate the street edge", cost="€", horizon="1-3 yrs",
                     agency="ULB (vending / land-use policy)", monitor="Active-frontage %; vendor count; evening activity",
                     action="Designate vending zones, enable ground-floor uses and open up blank frontages to bring people and 'eyes on the street'."),
    "seating": dict(title="Seating, parklets & rest points", cost="€", horizon="0-1 yrs",
                    agency="ULB / UUSDA public realm", monitor="Seats per 100 m; observed dwell time",
                    action="Add shaded seating and parklets at regular intervals, particularly near stops, markets and temples."),
    "tactical": dict(title="Pedestrian-priority pilot (tactical urbanism)", cost="€", horizon="0-1 yrs",
                     agency="ULB + traffic police", monitor="Pedestrian counts; vehicle diversion; business feedback",
                     action="SCREENING ONLY: low-speed, active street - test a timed pedestrian-priority or shared-street pilot with temporary materials."),
    "access": dict(title="Accessibility fixes (kerb ramps, tactile paving, clear width)", cost="€€", horizon="1-2 yrs",
                   agency="ULB / PWD", monitor="% crossings with ramps; clear width; accessibility audit rating",
                   action="Fix the audited accessibility barriers: ramps, tactile paving, minimum clear width, obstructions."),
}
SCENARIO_ACTIONS = {
    "footpath": "Complete footpaths", "calm": "Traffic calming (30 km/h)", "trees": "Street trees & shade",
    "light": "Pedestrian lighting", "seating": "Seating & parklets", "activate": "Frontage activation",
}


def scale(x, lo, hi):
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0, 1)


# -----------------------------------------------------------------------------------------
# 1. Segmentation + spatial context (heavy, run once per analysis)
# -----------------------------------------------------------------------------------------
def _sample_raster(xy, tif, crs):
    import rasterio
    with rasterio.open(tif) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        inv, rcrs = ~src.transform, src.crs
    pts = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs).to_crs(rcrs)
    cols, rows = inv * (pts.x.values, pts.y.values)
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full(len(xy), np.nan)
    out[ok] = arr[r[ok], c[ok]]
    return out


def build_segments(city, cells, junctions, seg_len=50, max_segments=15000):
    """Cut every street into ~seg_len m segments and attach raw spatial context."""
    G, crs = city.graph, city.crs
    edges = [(a, b, d) for a, b, d in G.edges(data=True) if d.get("highway") not in ("steps", "track")]
    total = sum(d["length"] for _, _, d in edges)
    if total / seg_len > max_segments:                       # keep big cities responsive
        seg_len = int(math.ceil(total / max_segments / 5) * 5)
    rows, geoms = [], []
    for a, b, d in sorted(edges, key=lambda t: (str(t[0]), str(t[1]))):
        geom = d.get("geometry")
        if geom is None:
            geom = LineString([(G.nodes[a]["x"], G.nodes[a]["y"]), (G.nodes[b]["x"], G.nodes[b]["y"])])
        L = geom.length
        if L < 1:
            continue
        n = max(1, int(round(L / seg_len)))
        sw = d.get("sidewalk")
        lit = d.get("lit")
        for k in range(n):
            seg = substring(geom, k * L / n, (k + 1) * L / n)
            rows.append(dict(edge_u=a, edge_v=b, k=k, n_in_edge=n, name=d.get("name") or "",
                             highway=d.get("highway", "residential"), speed=float(d.get("speed", 30)),
                             speed_tagged=bool(d.get("speed_tagged", False)), lanes=float(d.get("lanes", 1)),
                             sidewalk=np.nan if sw is None else float(bool(sw)),
                             lit=np.nan if lit is None else float(bool(lit)),
                             surface=d.get("surface") or "", length_m=seg.length))
            geoms.append(seg)
    seg = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)
    seg["seg_id"] = [f"S{i + 1:05d}" for i in range(len(seg))]
    n = len(seg)
    lines = seg.geometry.values
    mids = shapely.line_interpolate_point(lines, .5, normalized=True)
    mxy = np.c_[shapely.get_x(mids), shapely.get_y(mids)]
    ll = gpd.GeoSeries(mids, crs=crs).to_crs(4326)
    seg["lat"], seg["lon"] = ll.y.values.round(6), ll.x.values.round(6)
    buf = shapely.buffer(lines, 15)
    barea = shapely.area(buf)

    # canopy within 15 m
    cg = city.canopy.geometry.values if city.canopy is not None and len(city.canopy) else []
    seg["canopy15"] = 0.0
    if len(cg):
        t = shapely.STRtree(cg)
        ii, cj = t.query(buf, predicate="intersects")
        inter = shapely.area(shapely.intersection(buf[ii], np.asarray(cg)[cj]))
        seg["canopy15"] = np.clip(np.bincount(ii, weights=inter, minlength=n) / barea, 0, 1)

    # remote sensing at 3 points along the segment
    def along(tif):
        vals = []
        for f in (.25, .5, .75):
            p = shapely.line_interpolate_point(lines, f, normalized=True)
            vals.append(_sample_raster(np.c_[shapely.get_x(p), shapely.get_y(p)], tif, crs))
        return np.nanmean(np.vstack(vals), axis=0) if len(vals) else np.full(n, np.nan)
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg["ndvi"] = along(city.ndvi_tif) if city.ndvi_tif and os.path.exists(city.ndvi_tif) else np.nan
        seg["lst_c"] = along(city.lst_tif) if city.lst_tif and os.path.exists(city.lst_tif) else np.nan
    seg["lst_pct"] = pd.Series(seg["lst_c"]).rank(pct=True).values if seg["lst_c"].notna().any() else np.nan

    # active frontage proxy: destinations within 25 m of the street
    seg["doors"], seg["diversity"] = 0.0, 0.0
    if len(city.pois):
        pg = city.pois.geometry.values
        b25 = shapely.buffer(lines, 25)
        ii, pj = shapely.STRtree(pg).query(b25, predicate="intersects")
        seg["doors"] = np.bincount(ii, minlength=n)
        cat = city.pois["category"].values[pj]
        d = pd.DataFrame({"i": ii, "c": cat}).drop_duplicates()
        seg["diversity"] = d.groupby("i").size().reindex(range(n)).fillna(0).values
    seg["doors_per100"] = seg["doors"] / np.maximum(seg.length_m, 25) * 100

    # street furniture
    seg["benches"], seg["lamps"] = 0.0, 0.0
    fu = getattr(city, "furniture", None)
    if fu is not None and len(fu):
        b20 = shapely.buffer(lines, 20)
        for kind, col in (("bench", "benches"), ("lamp", "lamps")):
            g = fu[fu["kind"] == kind].geometry.values
            if len(g):
                ii, _ = shapely.STRtree(g).query(b20, predicate="intersects")
                seg[col] = np.bincount(ii, minlength=n)
    seg["lamps_per100"] = seg["lamps"] / np.maximum(seg.length_m, 25) * 100
    # park proximity
    seg["park_near"] = 0.0
    if len(city.parks):
        pt = shapely.STRtree(city.parks.geometry.values)
        _, dist = pt.query_nearest(mids, return_distance=True, all_matches=False)
        seg["park_near"] = (dist <= 100).astype(float)

    # junction risk at the ends of each edge (full weight for end segments, 40% inside)
    jr = dict(zip(junctions["node"], junctions["risk"])) if len(junctions) else {}
    ru = seg.edge_u.map(lambda x: jr.get(x, 0.0)).values
    rv = seg.edge_v.map(lambda x: jr.get(x, 0.0)).values
    first, last = (seg.k.values == 0), (seg.k.values == seg.n_in_edge.values - 1)
    near = np.where(first, ru, 0.0)
    near = np.maximum(near, np.where(last, rv, 0.0))
    seg["junc_risk_end"] = np.maximum(near, .4 * np.maximum(ru, rv))

    # demand + cell context
    from scipy.spatial import cKDTree
    poi_xy = np.c_[city.pois.geometry.x, city.pois.geometry.y] if len(city.pois) else np.zeros((0, 2))
    st_xy = np.c_[city.stops.geometry.x, city.stops.geometry.y] if len(city.stops) else np.zeros((0, 2))
    poi150 = np.array([len(x) for x in cKDTree(poi_xy).query_ball_point(mxy, 150)]) if len(poi_xy) else np.zeros(n)
    st200 = np.array([len(x) for x in cKDTree(st_xy).query_ball_point(mxy, 200)]) if len(st_xy) else np.zeros(n)
    mg = gpd.GeoDataFrame({"i": range(n)}, geometry=mids, crs=crs)
    keep = [c for c in ("cell_id", "walk", "transit", "green", "heat", "pop", "deprivation") if c in cells.columns]
    j = gpd.sjoin(mg, cells[keep + ["geometry"]], how="left", predicate="within").drop_duplicates("i").sort_values("i")
    for c in keep:
        seg[f"cell_{c}"] = j[c].values
    popd = np.nan_to_num(j["pop"].values) / (cells.geometry.area.iloc[0] / 1e6)
    seg["demand"] = np.clip(.4 * scale(poi150, 0, 25) + .3 * scale(st200, 0, 2) + .3 * scale(popd, 500, 10000), 0, 1)

    # corridor key (street name; unnamed streets grouped by type + 500 m block)
    blk = (np.floor(mxy[:, 0] / 500).astype(int).astype(str) + "_" + np.floor(mxy[:, 1] / 500).astype(int).astype(str))
    seg["corridor"] = np.where(seg.name != "", seg.name, "Unnamed " + seg.highway + " [" + blk + "]")
    seg.attrs["seg_len"] = seg_len
    return seg


# -----------------------------------------------------------------------------------------
# 2. Scoring (cheap; re-run with audit data / weights)
# -----------------------------------------------------------------------------------------
def _nm(rows):
    """Row-wise nanmean without 'mean of empty slice' warnings (all-NaN -> NaN)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.vstack(rows), axis=0)
    return out


def _traffic(speed, lanes, jr, vehicular):
    sev = u.ped_injury_risk(np.asarray(speed, float))
    widthf = .6 + .4 * np.minimum(1, np.asarray(lanes, float) * 3.3 / 13)
    risk = .6 * sev * widthf + .4 * np.asarray(jr, float) / 100
    return np.where(vehicular, 100 * (1 - np.clip(risk / .8, 0, 1)), 100.0)


def score_segments(seg, weights=None, audit=None, params=None):
    """Compute sub-scores, evidence basis, HSI, confidence, priority and intervention flags."""
    P = {**PARAMS, **(params or {})}
    W = {**DEFAULT_WEIGHTS, **(weights or {})}
    S = seg.copy()
    n = len(S)
    A = (audit.reindex(S.seg_id.values) if audit is not None and len(audit) else
         pd.DataFrame(index=S.seg_id.values))
    a = lambda col: A[col].values.astype(float) if col in A else np.full(n, np.nan)
    hw = S.highway.values
    ped = np.isin(hw, list(PED_HW))
    vehicular = ~np.isin(hw, ["footway", "path", "pedestrian", "cycleway", "steps"])
    sc, ba = {}, {}

    def finish(name, val, basis, audit_val=None):
        val = np.asarray(val, float).copy()
        basis = np.asarray(basis, object).copy()
        if audit_val is not None:
            m = np.isfinite(audit_val)
            val[m], basis[m] = audit_val[m], "audit"
        sc[name], ba[name] = val, basis

    # --- infrastructure
    sw = S.sidewalk.values
    base = np.where(ped, 80, np.where(sw == 1, 70, np.where(sw == 0, 15, P["unknown_sidewalk_score"])))
    sfc = S.surface.astype(str).str.lower().values
    surf = np.where(np.isin(sfc, list(PAVED)), 1.0, np.where(np.isin(sfc, list(UNPAVED)), .2, .5))
    val = .85 * base + .15 * 100 * surf
    basis = np.where(ped | ~np.isnan(sw), "osm", "default")
    wd = np.clip((a("a_clear_width_m") - P["width_poor_m"]) / (P["width_good_m"] - P["width_poor_m"]), 0, 1) * 100
    comp = np.vstack([(a("a_continuity") - 1) / 4 * 100, wd, (a("a_surface_condition") - 1) / 4 * 100,
                      100 - a("a_obstructions") / 3 * 100, 100 - a("a_parking_encroachment") / 3 * 100])
    au = _nm(comp)
    au = np.where(a("a_footpath_present") == 0, 10.0, au)
    au[~np.isfinite(au)] = np.nan
    finish("infra", val, basis, au)

    # --- traffic exposure
    speed = np.where(np.isfinite(a("a_vehicle_speed_kmh")), a("a_vehicle_speed_kmh"), S.speed.values)
    jr = S.junc_risk_end.values * np.where(a("a_crossing_available") == 1, .7, 1.0)
    val = _traffic(speed, S.lanes.values, jr, vehicular)
    basis = np.where(~vehicular | S.speed_tagged.values, "osm", "default")
    audit_t = np.where(np.isfinite(a("a_vehicle_speed_kmh")) | np.isfinite(a("a_crossing_available")), val, np.nan)
    finish("traffic", val, basis, audit_t)

    # --- shade / thermal comfort
    has_ndvi, has_lst = S.ndvi.notna().values, S.lst_pct.notna().values
    veg = np.where(has_ndvi, .5 * scale(S.canopy15, 0, .3) + .5 * scale(S.ndvi.fillna(0), .15, .6), scale(S.canopy15, 0, .3))
    val = 100 * np.where(has_lst, .7 * veg + .3 * (1 - S.lst_pct.fillna(.5).values), veg)
    basis = np.where(has_ndvi | has_lst, "remote", "proxy")
    finish("shade", val, basis, np.clip(a("a_shade_pct") / P["shade_target_pct"], 0, 1) * 100)

    # --- lighting
    lit, lamp = S.lit.values, scale(S.lamps_per100, 0, 3) * 100
    val = np.where(lit == 1, np.maximum(75, lamp), np.where(lit == 0, np.minimum(20, lamp),
                   np.where(S.lamps_per100 > 0, lamp, P["unknown_lighting_score"])))
    basis = np.where(~np.isnan(lit), "osm", np.where(S.lamps_per100 > 0, "proxy", "default"))
    finish("lighting", val, basis, (a("a_lighting_quality") - 1) / 4 * 100)

    # --- vitality
    vit = 100 * (.7 * scale(S.doors_per100, 0, 6) + .3 * scale(S.diversity, 0, 3))
    finish("vitality", vit, np.full(n, "proxy", object), np.clip(a("a_active_frontage_pct") / P["frontage_target_pct"], 0, 1) * 100)

    # --- stayability
    stay = 100 * (.5 * scale(S.benches / np.maximum(S.length_m, 25) * 100, 0, 2) + .3 * S.park_near + .2 * vit / 100)
    seat100 = a("a_seating_count") / np.maximum(S.length_m.values, 25) * 100
    finish("stay", stay, np.full(n, "proxy", object),
           100 * (.7 * np.clip(seat100 / P["seating_per_100m_target"], 0, 1) + .3 * S.park_near.values))

    # --- audit-only indicators (NaN -> excluded from the index)
    pr = _nm([(a(c) - 1) / 4 * 100 for c in ("a_perceived_safety_day", "a_perceived_safety_night", "a_perceived_safety_women")])
    ac = _nm([a("a_kerb_ramps") * 100, a("a_tactile_paving") * 100, wd,
              100 - a("a_obstructions") / 3 * 100, (a("a_accessibility_overall") - 1) / 4 * 100])
    pr[~np.isfinite(pr)], ac[~np.isfinite(ac)] = np.nan, np.nan
    sc["safety_perc"], ba["safety_perc"] = pr, np.where(np.isfinite(pr), "audit", "")
    sc["access"], ba["access"] = ac, np.where(np.isfinite(ac), "audit", "")

    for k in INDICATORS:
        S[f"s_{k}"], S[f"b_{k}"] = np.round(sc[k], 1), ba[k]
    # non-scored indicative safety proxy (kept separate so it is never double counted)
    S["safety_proxy"] = np.round(.4 * S.s_lighting + .3 * S.s_vitality + .3 * S.s_traffic, 1)
    S["audited"] = np.any([ba[k] == "audit" for k in INDICATORS], axis=0)
    return compute_hsi(S, W)


def compute_hsi(S, weights=None):
    W = {**DEFAULT_WEIGHTS, **(weights or {})}
    S = S.copy()
    num = np.zeros(len(S)); den = np.zeros(len(S))
    share = {b: np.zeros(len(S)) for b in BASES}
    for k in INDICATORS:
        v = S[f"s_{k}"].values
        ok = np.isfinite(v)
        num += np.where(ok, W[k] * np.nan_to_num(v), 0)
        den += np.where(ok, W[k], 0)
        for b in BASES:
            share[b] += np.where(ok & (S[f"b_{k}"].values == b), W[k], 0)
    S["hsi"] = np.round(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), 1)
    for b in BASES:
        S[f"pct_{b}"] = np.round(share[b] / np.maximum(den, 1e-9), 2)
    measured = S.pct_audit + S.pct_osm + S.pct_remote
    S["confidence"] = np.where(S.pct_audit >= .5, "High", np.where(measured >= .5, "Medium", "Low"))
    S["priority"] = np.round((100 - S.hsi) * (.4 + .6 * S.demand), 1)
    S["audit_priority"] = np.round(S.demand * (100 - S.hsi) / 100 * (1 - S.pct_audit), 3)
    # intervention flags
    vehicular = ~np.isin(S.highway.values, ["footway", "path", "pedestrian", "cycleway", "steps"])
    low_traffic_hw = np.isin(S.highway.values, ["residential", "living_street", "service", "unclassified"])
    flags = {
        "footpath": S.s_infra < 40,
        "calm": (S.s_traffic < 45) & vehicular & (S.speed >= 35),
        "trees": S.s_shade < 35,
        "light": S.s_lighting < 40,
        "activate": (S.s_vitality < 30) & (S.demand >= .4),
        "seating": (S.s_stay < 30) & (S.demand >= .4),
        "tactical": vehicular & low_traffic_hw & (S.speed <= 30) & (S.s_vitality >= 50) & (S.length_m >= 40),
        "access": S.s_access < 40,
    }
    FL = pd.DataFrame({c: np.asarray(v, bool) for c, v in flags.items()})
    S["iv"] = [",".join(FL.columns[row]) for row in FL.values]
    return S


# -----------------------------------------------------------------------------------------
# 3. Corridors & recommendations
# -----------------------------------------------------------------------------------------
def corridors(S, min_len=150):
    rows = []
    for name, g in S.groupby("corridor"):
        L = g.length_m.sum()
        if L < min_len:
            continue
        w = g.length_m / L
        r = dict(corridor=name, length_m=round(L), segments=len(g), hsi=round(float((g.hsi * w).sum()), 1),
                 demand=round(float((g.demand * w).sum()), 2),
                 priority_score=round(float((g.priority * g.length_m).sum() / 1000), 2),
                 confidence_high_pct=round(100 * float((g.confidence.eq("High") * w).sum())),
                 audited_pct=round(100 * float((g.audited * w).sum())))
        for code in INTERVENTIONS:
            r[f"pct_{code}"] = round(100 * float((g.iv.str.contains(rf"\b{code}\b", regex=True) * w).sum()))
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True) if len(df) else df


def corridor_recommendations(C, top=10, min_pct=30):
    rows = []
    for _, r in C.head(top).iterrows():
        items = sorted(((r[f"pct_{c}"], c) for c in INTERVENTIONS if r[f"pct_{c}"] >= min_pct), reverse=True)
        for rank, (pct, c) in enumerate(items[:4], 1):
            iv = INTERVENTIONS[c]
            rows.append(dict(corridor=r.corridor, corridor_rank=int(C.index[C.corridor == r.corridor][0]) + 1, rank=rank,
                             intervention=iv["title"], code=c, share_of_length_pct=int(pct), action=iv["action"],
                             cost_band=iv["cost"], horizon=iv["horizon"], typical_lead_agency=iv["agency"] + " (verify locally)",
                             monitoring_indicator=iv["monitor"], length_m=int(r.length_m), corridor_hsi=r.hsi,
                             evidence_confidence=("audited" if r.audited_pct >= 50 else "open-data only - audit before design")))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------------------
# 4. Before / after simulator  (MODELLED, assumption-driven)
# -----------------------------------------------------------------------------------------
DEFAULT_EFFECTS = dict(footpath_target=80, calm_speed=30, trees_gain=30, light_target=80, seating_gain=25, activate_gain=20)


def simulate(S, corridor_names, actions, effects=None, weights=None, unit_costs=None):
    """Apply assumed effects of `actions` to segments on the chosen corridors; recompute HSI."""
    E = {**DEFAULT_EFFECTS, **(effects or {})}
    T = S.copy()
    sel = T.corridor.isin(corridor_names).values
    veh = ~np.isin(T.highway.values, ["footway", "path", "pedestrian", "cycleway", "steps"])
    ch = {a: np.zeros(len(T), bool) for a in SCENARIO_ACTIONS}
    if "footpath" in actions:
        m = sel & (T.s_infra < E["footpath_target"]).values
        T.loc[m, "s_infra"] = E["footpath_target"]; ch["footpath"] = m
    if "calm" in actions:
        m = sel & veh & (T.speed > E["calm_speed"]).values
        new = _traffic(np.minimum(T.speed.values, E["calm_speed"]), T.lanes.values, T.junc_risk_end.values * .7, veh)
        T.loc[m, "s_traffic"] = np.maximum(T.s_traffic.values, new)[m]; ch["calm"] = m
    if "trees" in actions:
        m = sel & (T.s_shade < 70).values
        T.loc[m, "s_shade"] = np.minimum(100, T.s_shade[m] + E["trees_gain"]); ch["trees"] = m
    if "light" in actions:
        m = sel & (T.s_lighting < E["light_target"]).values
        T.loc[m, "s_lighting"] = E["light_target"]; ch["light"] = m
    if "seating" in actions:
        m = sel & (T.s_stay < 80).values
        T.loc[m, "s_stay"] = np.minimum(100, T.s_stay[m] + E["seating_gain"]); ch["seating"] = m
    if "activate" in actions:
        m = sel & (T.s_vitality < 70).values
        T.loc[m, "s_vitality"] = np.minimum(100, T.s_vitality[m] + E["activate_gain"]); ch["activate"] = m
    T = compute_hsi(T, weights)
    w = T.length_m
    wm = lambda D, col: float((D[col] * w).sum() / w.sum())
    res = dict(hsi_before=wm(S, "hsi"), hsi_after=wm(T, "hsi"),
               poor_before=float(100 * (w * (S.hsi < 40)).sum() / w.sum()), poor_after=float(100 * (w * (T.hsi < 40)).sum() / w.sum()),
               segments_changed=int(np.any(list(ch.values()), axis=0).sum()),
               km_treated={a: float(T.length_m[ch[a]].sum() / 1000) for a in actions})
    res["delta"] = res["hsi_after"] - res["hsi_before"]
    cost = None
    if unit_costs and any(unit_costs.get(a) for a in actions):
        cost = sum(res["km_treated"][a] * float(unit_costs.get(a) or 0) for a in actions)
    res["cost"] = cost
    return T, res
