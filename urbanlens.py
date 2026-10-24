#!/usr/bin/env python3
"""
UrbanLens - urban analytics for walkability, heat vulnerability, junction safety,
public transport connectivity, green-space access and neighbourhood equity.

Pipeline:  data (OSM / synthetic)  ->  250 m analysis grid  ->  network-based indicators
           -> remote-sensing fusion (NDVI / LST GeoTIFFs)  ->  equity analysis
           -> AI layer (typology clustering + rule-based recommendation engine
              + optional LLM planning brief)  ->  interactive map + HTML report.

Usage
    python urbanlens.py demo --out out_demo
    python urbanlens.py osm "Kirkwall, Scotland" --out out_kirkwall [--socio zones.gpkg] \
                        [--ndvi ndvi.tif --lst lst.tif] [--llm]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CATS = ["food", "shop", "education", "health", "services", "leisure"]
SCORES = ["walk", "safety", "comfort", "transit", "green", "heat"]
SCORE_LABELS = {
    "walk": "Walkability & street accessibility",
    "safety": "Junction safety",
    "comfort": "Pedestrian comfort",
    "transit": "Public transport connectivity",
    "green": "Green-space access",
    "heat": "Heat & climate resilience",
}
DEFAULT_WEIGHTS = dict(walk=.20, safety=.20, comfort=.15, transit=.15, green=.15, heat=.15)

DEFAULT_SPEED = {  # km/h, used only when OSM maxspeed is missing - edit for your country
    "motorway": 100, "trunk": 80, "primary": 50, "secondary": 50, "tertiary": 40,
    "unclassified": 40, "residential": 30, "living_street": 10, "service": 20,
    "footway": 0, "path": 0, "pedestrian": 0, "steps": 0, "cycleway": 0, "track": 0,
}
ROAD_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
             "residential", "service", "living_street", "pedestrian", "footway", "path", "cycleway", "steps", "track"]
ARTERIAL = {"motorway", "trunk", "primary", "secondary"}
NONVEHICLE = {"footway", "path", "pedestrian", "steps", "cycleway", "track"}

# Pedestrian severe-injury risk vs. vehicle impact speed (approx. after Tefft 2013, AIS4+).
_SPD = [0, 20, 26, 37, 50, 63, 74, 130]
_RSK = [0, .03, .10, .25, .50, .75, .90, 1.0]


def ped_injury_risk(kmh):
    return np.interp(kmh, _SPD, _RSK)


def scale(x, lo, hi):
    """Linear 0-1 ramp; works if lo > hi (inverse ramp)."""
    x = np.asarray(x, float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


# ----------------------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------------------
@dataclass
class CityData:
    name: str
    crs: str
    graph: nx.Graph                    # undirected; node: x,y,street_count; edge: length,highway,speed,lanes,sidewalk
    pois: gpd.GeoDataFrame             # category in CATS
    stops: gpd.GeoDataFrame            # mode in {bus,tram,rail}, trips_per_hr
    crossings: gpd.GeoDataFrame        # kind in {signal, marked}
    parks: gpd.GeoDataFrame            # public green space polygons
    buildings: gpd.GeoDataFrame
    canopy: gpd.GeoDataFrame           # tree-canopy polygons (may be empty)
    boundary: Polygon
    socio: gpd.GeoDataFrame | None = None   # polygons: population, deprivation (0-1), elderly_share
    ndvi_tif: str | None = None
    lst_tif: str | None = None
    pop_tif: str | None = None            # e.g. WorldPop persons-per-pixel GeoTIFF
    provenance: dict = field(default_factory=dict)
    furniture: gpd.GeoDataFrame | None = None   # points: kind in {bench, lamp}


# ----------------------------------------------------------------------------------------
# Synthetic demo city (offline, reproducible) - exercises exactly the same pipeline
# ----------------------------------------------------------------------------------------
def synthetic_city(out_dir: str, seed: int = 7) -> CityData:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    crs = "EPSG:32630"
    X0, Y0, W = 697000.0, 5707500.0, 5000.0
    cx, cy = X0 + W / 2, Y0 + W / 2

    def dcen(x, y):
        return np.hypot(x - cx, y - cy) / (W / 2)

    # --- street network on a jittered grid, thinned towards the periphery -----------------
    n = 34
    sp = W / (n - 1)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j), x=X0 + i * sp + rng.normal(0, 12), y=Y0 + j * sp + rng.normal(0, 12))
    arterial_idx = {6, 17, 28}
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= n or b[1] >= n:
                    continue
                art = (j in arterial_idx) if di == 1 else (i in arterial_idx)
                G.add_edge(a, b, art=art,
                           name=f"{'Arterial' if art else 'Street'} {'E-W' if di == 1 else 'N-S'} {j if di == 1 else i}")
    edges = list(G.edges(data=True))
    rng.shuffle(edges)
    for a, b, d in edges:  # thin periphery to create cul-de-sac / low-permeability fringe
        if d["art"]:
            continue
        xm = (G.nodes[a]["x"] + G.nodes[b]["x"]) / 2
        ym = (G.nodes[a]["y"] + G.nodes[b]["y"]) / 2
        if rng.random() < 0.62 * dcen(xm, ym) ** 2:
            G.remove_edge(a, b)
            if not nx.is_connected(G):
                G.add_edge(a, b, **d)
    for a, b, d in G.edges(data=True):
        xa, ya, xb, yb = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[b]["x"], G.nodes[b]["y"]
        xm, ym = (xa + xb) / 2, (ya + yb) / 2
        dc = dcen(xm, ym)
        d["length"] = float(np.hypot(xa - xb, ya - yb))
        d["lit"] = bool(rng.random() < (.92 if d["art"] else max(.15, .8 - .7 * dc)))
        d["surface"] = "asphalt" if (d["art"] or dc < .6 or rng.random() < .5) else "unpaved"
        d["speed_tagged"] = True
        if d["art"]:
            d.update(highway="primary" if (dc < .6) else "secondary", speed=50 if dc < .6 else 60,
                     lanes=int(rng.choice([2, 4], p=[.5, .5])), sidewalk=bool(rng.random() < .9))
        else:
            d.update(highway="residential", speed=30 if (dc < .45 and rng.random() < .8) else 50, lanes=2,
                     sidewalk=bool(rng.random() < (0.97 - 0.5 * dc ** 2)))
    G = nx.relabel_nodes(G, {k: i for i, k in enumerate(G.nodes)})
    for nd in G.nodes:
        G.nodes[nd]["street_count"] = G.degree[nd]
    nodes_xy = np.array([[G.nodes[k]["x"], G.nodes[k]["y"]] for k in G.nodes])

    # --- crossings (deliberately patchy) -------------------------------------------------
    cr = []
    for nd in G.nodes:
        if G.degree[nd] < 3:
            continue
        inc = [G.edges[nd, m] for m in G[nd]]
        art = any(e["art"] for e in inc)
        r = rng.random()
        if art and r < .35:
            cr.append(("signal", nd))
        elif r < (.75 if art else .30):
            cr.append(("marked", nd))
    crossings = gpd.GeoDataFrame({"kind": [k for k, _ in cr]},
                                 geometry=[Point(nodes_xy[nd] + rng.normal(0, 6, 2)) for _, nd in cr], crs=crs)

    # --- POIs ------------------------------------------------------------------------------
    hubs = np.array([[cx, cy], [cx - 1100, cy + 900], [cx + 1000, cy - 700]])
    pts = []
    while len(pts) < 1000:
        h = hubs[rng.choice(3, p=[.6, .2, .2])]
        p = h + rng.normal(0, 1150 if (h == hubs[0]).all() else 550, 2)
        if X0 < p[0] < X0 + W and Y0 < p[1] < Y0 + W:
            pts.append(p)
    pts = np.array(pts)
    pois = gpd.GeoDataFrame({"category": rng.choice(CATS, len(pts), p=[.3, .25, .08, .07, .15, .15])},
                            geometry=[Point(p) for p in pts], crs=crs)

    # --- transit stops -----------------------------------------------------------------------
    stops = []
    for nd in G.nodes:
        inc = [G.edges[nd, m] for m in G[nd]]
        if any(e["art"] for e in inc) and rng.random() < .5:
            xy = nodes_xy[nd]
            dc = dcen(*xy)
            stops.append(("bus", float(max(2, rng.normal(14 - 9 * dc, 2))), Point(xy)))
    for yy in (cy - 150, cy + 400):  # one tram line along a corridor (centre, W-E)
        pass
    for xx in np.arange(X0 + 800, X0 + W - 700, 500):
        stops.append(("tram", 10.0, Point(xx, cy + 20)))
    for xx in (cx - 700, cx + 800):
        stops.append(("rail", 8.0, Point(xx, cy - 60)))
    stops = gpd.GeoDataFrame({"mode": [s[0] for s in stops], "trips_per_hr": [s[1] for s in stops]},
                             geometry=[s[2] for s in stops], crs=crs)

    # --- parks (fewer in the south-east to bake in an equity story) -------------------------
    parks = []
    for _ in range(40):
        p = np.array([X0 + rng.random() * W, Y0 + rng.random() * W])
        se = ((p[0] - X0) / W - (p[1] - Y0) / W)  # >0 toward south-east
        if rng.random() < scale(se, .2, 1.0):
            continue
        r = rng.uniform(35, 150)
        parks.append(Point(p).buffer(r, 16))
        if len(parks) >= 16:
            break
    parks = gpd.GeoDataFrame({"name": [f"Park {i+1}" for i in range(len(parks))]}, geometry=parks, crs=crs)

    # --- buildings & canopy ---------------------------------------------------------------
    bl = []
    for _ in range(26000):
        p = np.array([X0 + rng.random() * W, Y0 + rng.random() * W])
        if rng.random() < dcen(*p) ** 1.3 * .85:
            continue
        w, h = rng.uniform(12, 35, 2)
        bl.append(box(p[0], p[1], p[0] + w, p[1] + h))
    bld = gpd.GeoDataFrame(geometry=bl, crs=crs)
    park_union = parks.union_all()
    bld = bld[~bld.intersects(park_union)].reset_index(drop=True)
    tp = []
    for _ in range(38000):
        p = np.array([X0 + rng.random() * W, Y0 + rng.random() * W])
        se = ((p[0] - X0) / W - (p[1] - Y0) / W)
        if rng.random() < scale(se, -.2, .9) * .85:
            continue
        tp.append(Point(p).buffer(rng.uniform(4, 8), 6))
    canopy = gpd.GeoDataFrame(geometry=tp, crs=crs)

    # --- street furniture: benches cluster near destinations/parks, lamps follow lit streets ----
    bpts = []
    for p in list(pts[rng.choice(len(pts), 160, replace=False)]) + [np.array([g.centroid.x, g.centroid.y]) for g in parks.geometry]:
        bpts.append(Point(p + rng.normal(0, 12, 2)))
    lpts = []
    for a, b, d in G.edges(data=True):
        if d.get("lit"):
            A, B = nodes_xy[a], nodes_xy[b]
            for t in np.linspace(0, 1, max(1, int(d["length"] / 45)), endpoint=False) + .1:
                lpts.append(Point(A + (B - A) * t + rng.normal(0, 3, 2)))
    furniture = gpd.GeoDataFrame({"kind": ["bench"] * len(bpts) + ["lamp"] * len(lpts)}, geometry=bpts + lpts, crs=crs)

    boundary = box(X0, Y0, X0 + W, Y0 + W)

    # --- socio-economic zones (500 m), to exercise areal interpolation -----------------------
    zs = 500
    zones = []
    for x in np.arange(X0, X0 + W, zs):
        for y in np.arange(Y0, Y0 + W, zs):
            p = np.array([x + zs / 2, y + zs / 2])
            rel = ((p[0] - X0) / W - (p[1] - Y0) / W)
            dep = float(np.clip(.45 + .5 * rel + rng.normal(0, .08), 0, 1))
            pop = float(max(80, (1 - .75 * dcen(*p)) * 3600 + rng.normal(0, 250)))
            eld = float(np.clip(.12 + .05 * (dep - .5) + rng.normal(0, .03), .03, .3))
            zones.append((pop, dep, eld, box(x, y, x + zs, y + zs)))
    socio = gpd.GeoDataFrame({"population": [z[0] for z in zones], "deprivation": [z[1] for z in zones],
                              "elderly_share": [z[2] for z in zones]}, geometry=[z[3] for z in zones], crs=crs)

    # --- synthetic "remote sensing" rasters, written as real GeoTIFFs ----------------------
    res = 30.0
    nx_ = int(W / res)
    tr = from_origin(X0, Y0 + W, res, res)
    shp = (nx_, nx_)

    def rz(gdf):
        return rasterize([(g, 1) for g in gdf.geometry], out_shape=shp, transform=tr, fill=0).astype("float32")

    r_b, r_c, r_p = rz(bld), rz(canopy), rz(parks)
    r_b, r_c, r_p = [gaussian_filter(a, 1.2) for a in (r_b, r_c, r_p)]
    ndvi = np.clip(.18 + .9 * r_c + .6 * r_p - .45 * r_b + rng.normal(0, .03, shp), -.05, .9).astype("float32")
    yy, xx = np.mgrid[0:nx_, 0:nx_]
    cdist = np.hypot(xx - nx_ / 2, yy - nx_ / 2) / (nx_ / 2)
    lst = (31 + 5.5 * (1 - cdist * .6) * (r_b > .08) + 3 * gaussian_filter(r_b, 6) * 6 - 7 * ndvi
           + rng.normal(0, .4, shp)).astype("float32")
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for nm, arr in (("ndvi", ndvi), ("lst", lst)):
        pth = os.path.join(out_dir, f"synthetic_{nm}.tif")
        with rasterio.open(pth, "w", driver="GTiff", height=nx_, width=nx_, count=1, dtype="float32",
                           crs=crs, transform=tr) as dst:
            dst.write(arr, 1)
        paths[nm] = pth

    return CityData("Synthetic demo city (NOT a real place)", crs, G, pois, stops, crossings, parks, bld, canopy,
                    boundary, socio, paths["ndvi"], paths["lst"], None,
                    {"source": "synthetic generator (seeded)", "synthetic": True,
                     "note": "Demo data. Patterns (e.g. deprived south-east) are baked in to show the workflow."},
                    furniture=furniture)


# ----------------------------------------------------------------------------------------
# Real-world loader (OpenStreetMap via osmnx). Needs internet access.
# ----------------------------------------------------------------------------------------
def _first(v):
    return v[0] if isinstance(v, (list, tuple)) else v


def _parse_speed(v, hw):
    try:
        s = str(_first(v)).lower()
        val = float(re.findall(r"[\d.]+", s)[0])
        return val * 1.609 if "mph" in s else val
    except Exception:
        return float(DEFAULT_SPEED.get(hw, 30))


def _parse_lanes(v, hw):
    try:
        return int(float(re.findall(r"[\d.]+", str(_first(v)))[0]))
    except Exception:
        return 2 if hw in ARTERIAL else 1


def place_area_km2(place: str):
    """Geocode a place name and return (area_km2, (lon, lat) centroid). Needs internet."""
    import osmnx as ox
    g = ox.geocode_to_gdf(place)
    area = float(g.to_crs(g.estimate_utm_crs()).area.iloc[0] / 1e6)
    c = g.geometry.iloc[0].centroid
    return area, (c.x, c.y)


def coverage_report(city: CityData) -> pd.DataFrame:
    """How much OSM data was found - lets users judge whether results are trustworthy."""
    G = city.graph
    rows = [("Street segments", G.number_of_edges(), 200, "Very sparse street data: results unreliable"),
            ("Junctions (3+ legs)", sum(1 for n in G.nodes if G.degree[n] >= 3), 30, "Few junctions mapped"),
            ("Shops / services / amenities", len(city.pois), 40, "Few destinations mapped - walkability will be understated"),
            ("Transit stops", len(city.stops), 5, "Few/no stops mapped - transit score unreliable"),
            ("Marked crossings / signals", len(city.crossings), 10, "Crossings rarely mapped - safety scores pessimistic"),
            ("Parks & green spaces", len(city.parks), 3, "Few green spaces mapped"),
            ("Buildings", len(city.buildings), 300, "Few buildings - population proxy & heat exposure weak"),
            ("Mapped trees", len(city.canopy), 50, "Tree data sparse - use NDVI raster for vegetation")]
    sw = [d.get("sidewalk") for *_, d in G.edges(data=True) if d.get("highway") not in NONVEHICLE]
    known = sum(1 for v in sw if v is not None)
    rows.append(("Roads with a sidewalk tag (%)", round(100 * known / len(sw), 1) if sw else 0, 15,
                 "Sidewalks mostly untagged - infrastructure scores rest on an assumed default; field-audit them"))
    ln = [d.get("lit") for *_, d in G.edges(data=True) if d.get("highway") not in NONVEHICLE]
    rows.append(("Roads with a lighting tag (%)", round(100 * sum(1 for v in ln if v is not None) / len(ln), 1) if ln else 0, 10,
                 "Lighting rarely tagged - lighting score uses lamp points or an assumed default"))
    df = pd.DataFrame(rows, columns=["Layer", "Found", "Min. for reliable", "If low"])
    df["Status"] = np.where(df["Found"] >= df["Min. for reliable"], "OK", "LOW")
    return df[["Layer", "Found", "Status", "If low"]]


def load_osm(place: str, socio_path: str | None = None, ndvi: str | None = None, lst: str | None = None,
             pop: str | None = None, bbox: str | None = None) -> CityData:
    import osmnx as ox

    ox.settings.use_cache = True
    # osmnx drops most tags by default - ask for the ones the street-level scoring needs
    extra = ["sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right", "lit", "surface", "footway", "smoothness"]
    ox.settings.useful_tags_way = list(dict.fromkeys(list(ox.settings.useful_tags_way) + extra))
    bb = tuple(float(v) for v in bbox.split(",")) if bbox else None   # west,south,east,north
    G0 = (ox.graph_from_bbox(bb, network_type="all", simplify=True) if bb
          else ox.graph_from_place(place, network_type="all", simplify=True))
    G0 = ox.project_graph(G0)
    crs = str(G0.graph["crs"])
    G = nx.Graph()
    for u, d in G0.nodes(data=True):
        G.add_node(u, x=d["x"], y=d["y"])
    for u, v, d in G0.edges(data=True):
        hw = _first(d.get("highway", "residential"))
        hw = min([h for h in (d["highway"] if isinstance(d["highway"], list) else [hw])],
                 key=lambda h: ROAD_RANK.index(h) if h in ROAD_RANK else 99)
        if hw in NONVEHICLE:
            side = True
        else:
            sws = [str(_first(d.get(k, ""))).lower() for k in ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right")]
            if any(v in ("both", "left", "right", "yes", "separate") for v in sws) or hw in ("living_street", "pedestrian"):
                side = True
            elif any(v in ("no", "none") for v in sws):
                side = False
            else:
                side = None                      # unknown - NOT the same as "no sidewalk"
        lit_raw = str(_first(d.get("lit", ""))).lower()
        lit = True if lit_raw in ("yes", "24/7", "automatic", "sunset-sunrise") else (False if lit_raw == "no" else None)
        try:
            wd = float(re.findall(r"[\d.]+", str(_first(d.get("width"))))[0])
        except Exception:
            wd = None
        attrs = dict(length=float(d["length"]), highway=hw, speed=_parse_speed(d.get("maxspeed"), hw),
                     lanes=_parse_lanes(d.get("lanes"), hw), sidewalk=side, art=hw in ARTERIAL,
                     speed_tagged=d.get("maxspeed") is not None, name=str(_first(d.get("name")) or ""),
                     lit=lit, surface=_first(d.get("surface")), width=wd, geometry=d.get("geometry"))
        if G.has_edge(u, v) and G.edges[u, v]["length"] <= attrs["length"]:
            continue
        G.add_edge(u, v, **attrs)
    for nd in G.nodes:
        G.nodes[nd]["street_count"] = G.degree[nd]

    def feats(tags):
        try:
            g = (ox.features_from_bbox(bb, tags=tags) if bb else ox.features_from_place(place, tags=tags)).to_crs(crs)
        except Exception:
            g = gpd.GeoDataFrame(geometry=[], crs=crs)
        return g[~g.geometry.is_empty]

    tags = {"food": {"amenity": ["restaurant", "cafe", "fast_food", "pub", "bar"]},
            "shop": {"shop": True},
            "education": {"amenity": ["school", "kindergarten", "college", "university"]},
            "health": {"amenity": ["clinic", "doctors", "pharmacy", "hospital", "dentist"]},
            "services": {"amenity": ["post_office", "bank", "library", "community_centre", "townhall"]},
            "leisure": {"leisure": ["fitness_centre", "sports_centre", "playground"], "amenity": ["cinema", "theatre"]}}
    parts = []
    for cat, t in tags.items():
        g = feats(t)
        if len(g):
            g = g.assign(category=cat)
            g["geometry"] = g.geometry.representative_point()
            parts.append(g[["category", "geometry"]])
    pois = pd.concat(parts) if parts else gpd.GeoDataFrame({"category": []}, geometry=[], crs=crs)
    pois = gpd.GeoDataFrame(pois, crs=crs)

    s = feats({"highway": ["bus_stop"], "railway": ["tram_stop", "station", "halt", "subway_entrance"],
               "public_transport": ["stop_position"]})
    if len(s):
        s = s.copy()
        s["geometry"] = s.geometry.representative_point()
        rail = s.get("railway", pd.Series(index=s.index, dtype=object)).isin(["station", "halt", "subway_entrance"])
        tram = s.get("railway", pd.Series(index=s.index, dtype=object)).eq("tram_stop")
        s["mode"] = np.where(rail, "rail", np.where(tram, "tram", "bus"))
        s["trips_per_hr"] = s["mode"].map({"bus": 6.0, "tram": 8.0, "rail": 6.0})  # PLACEHOLDER: supply GTFS for real headways
        stops = gpd.GeoDataFrame(s[["mode", "trips_per_hr"]], geometry=s.geometry, crs=crs)
    else:
        stops = gpd.GeoDataFrame({"mode": [], "trips_per_hr": []}, geometry=[], crs=crs)

    c = feats({"highway": ["crossing", "traffic_signals"], "crossing": True})
    if len(c):
        c = c.copy()
        c["geometry"] = c.geometry.representative_point()
        hwy = c.get("highway", pd.Series(index=c.index, dtype=object))
        c["kind"] = np.where(hwy.eq("traffic_signals"), "signal", "marked")
        crossings = gpd.GeoDataFrame(c[["kind"]], geometry=c.geometry, crs=crs)
    else:
        crossings = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=crs)

    pk = feats({"leisure": ["park", "garden", "nature_reserve", "recreation_ground"], "landuse": ["forest", "grass"],
                "natural": ["wood"]})
    pk = pk[pk.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].reset_index(drop=True)
    pk["name"] = [f"Green space {i+1}" for i in range(len(pk))]
    fu = feats({"amenity": ["bench"], "highway": ["street_lamp"]})
    if len(fu):
        fu = fu.copy()
        fu["geometry"] = fu.geometry.representative_point()
        hwy = fu.get("highway", pd.Series(index=fu.index, dtype=object))
        furniture = gpd.GeoDataFrame({"kind": np.where(hwy.eq("street_lamp"), "lamp", "bench")}, geometry=fu.geometry.values, crs=crs)
    else:
        furniture = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=crs)
    b = feats({"building": True})
    b = b[b.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].reset_index(drop=True)
    trees = feats({"natural": ["tree", "tree_row"]})
    canopy = gpd.GeoDataFrame(geometry=list(trees.geometry.buffer(4)) if len(trees) else [], crs=crs)

    if bb:
        boundary = gpd.GeoSeries([box(*bb)], crs=4326).to_crs(crs).iloc[0]
    else:
        boundary = ox.geocode_to_gdf(place).to_crs(crs).geometry.iloc[0]
    socio = None
    if socio_path:
        socio = gpd.read_file(socio_path).to_crs(crs)
    return CityData(place, crs, G, pois, stops, crossings, pk, b, canopy, boundary, socio, ndvi, lst, pop,
                    {"source": "OpenStreetMap via osmnx", "synthetic": False,
                     "note": "Transit frequencies are placeholders unless GTFS is integrated; "
                             "population is a building-area proxy unless --socio is supplied."},
                    furniture=furniture)


# ----------------------------------------------------------------------------------------
# Raster / vector helpers
# ----------------------------------------------------------------------------------------
def make_grid(boundary, crs, size):
    minx, miny, maxx, maxy = boundary.bounds
    cells = [box(x, y, x + size, y + size) for x in np.arange(minx, maxx, size) for y in np.arange(miny, maxy, size)]
    g = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=crs)
    g = g[g.geometry.centroid.within(boundary)].reset_index(drop=True)
    g["cell_id"] = range(len(g))
    return g


def area_share(cells, polys):
    """Share of each cell covered by (dissolved) polygons."""
    if polys is None or len(polys) == 0:
        return np.zeros(len(cells))
    u = shapely.union_all(polys.geometry.values)
    inter = shapely.intersection(cells.geometry.values, u)
    return shapely.area(inter) / cells.area.values


def zonal_mean(cells, tif):
    import rasterio
    with rasterio.open(tif) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        inv = ~src.transform
        cg = cells.to_crs(src.crs)
    out = np.full(len(cells), np.nan)
    for k, g in enumerate(cg.geometry.values):
        x0, y0, x1, y1 = g.bounds
        c0, r1 = inv * (x0, y0)
        c1, r0 = inv * (x1, y1)
        r0, r1 = int(max(0, np.floor(min(r0, r1)))), int(min(arr.shape[0], np.ceil(max(r0, r1))))
        c0, c1 = int(max(0, np.floor(min(c0, c1)))), int(min(arr.shape[1], np.ceil(max(c0, c1))))
        if r1 > r0 and c1 > c0:
            win = arr[r0:r1, c0:c1]
            if np.isfinite(win).any():
                out[k] = np.nanmean(win)
    return out


def zonal_sum(cells, tif, k=5):
    """Persons per cell from a persons-per-pixel raster (e.g. WorldPop).
    Samples k*k points per cell, converts pixel counts to density, then multiplies by cell area.
    Preserves totals and works whether pixels are finer or coarser than the grid."""
    import rasterio
    with rasterio.open(tif) as src:
        arr = src.read(1, masked=True).astype("float64").filled(0.0)
        arr[arr < 0] = 0
        inv, rcrs = ~src.transform, src.crs
        rx, ry = src.res
        if rcrs.is_geographic:
            lat = float(cells.to_crs(4326).geometry.centroid.y.mean())
            pix_area = (rx * 111320 * np.cos(np.radians(lat))) * (ry * 110540)
        else:
            pix_area = rx * ry * float(rcrs.linear_units_factor[1]) ** 2
    n = len(cells)
    b = cells.geometry.bounds
    off = (np.arange(k) + .5) / k
    X = b.minx.values[:, None] + off[None, :] * (b.maxx - b.minx).values[:, None]
    Y = b.miny.values[:, None] + off[None, :] * (b.maxy - b.miny).values[:, None]
    XX = (X[:, :, None] * np.ones((1, 1, k))).reshape(n, -1)
    YY = (Y[:, None, :] * np.ones((1, k, 1))).reshape(n, -1)
    pts = gpd.GeoSeries(gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=cells.crs).to_crs(rcrs)
    cols, rows = inv * (pts.x.values, pts.y.values)
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    dens = np.zeros(len(r))
    dens[ok] = arr[r[ok], c[ok]] / pix_area
    return dens.reshape(n, -1).sum(axis=1) * cells.area.values / (k * k)


def apportion_socio(cells, socio):
    pop = np.zeros(len(cells))
    dep = np.full(len(cells), np.nan)
    eld = np.full(len(cells), np.nan)
    s = socio.copy()
    s["src_area"] = s.area
    for col, dflt in (("deprivation", np.nan), ("elderly_share", np.nan)):
        if col not in s:
            s[col] = dflt
    inter = gpd.overlay(cells[["cell_id", "geometry"]], s[["population", "deprivation", "elderly_share", "src_area", "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["a"] = inter.area
    inter["pop_w"] = inter["population"] * inter["a"] / inter["src_area"]
    gb = inter.groupby("cell_id")
    pop[gb["pop_w"].sum().index.values] = gb["pop_w"].sum().values
    a = gb["a"].sum()
    for arr, col in ((dep, "deprivation"), (eld, "elderly_share")):
        wsum = (inter[col] * inter["a"]).groupby(inter["cell_id"]).sum() / a
        arr[wsum.index.values] = wsum.values
    return pop, dep, eld


# ----------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------
@dataclass
class Result:
    city: CityData
    cells: gpd.GeoDataFrame
    junctions: gpd.GeoDataFrame
    zones: list
    recs: pd.DataFrame
    equity: pd.DataFrame
    typologies: dict
    summary: dict
    weights: dict


def analyze(city: CityData, cell_size=250, weights=None, verbose=True) -> Result:
    log = (lambda *a: print("  ·", *a)) if verbose else (lambda *a: None)
    weights = weights or DEFAULT_WEIGHTS
    G = city.graph
    cells = make_grid(city.boundary, city.crs, cell_size)
    cent = np.array([[p.x, p.y] for p in cells.geometry.centroid])
    cell_area = cell_size ** 2
    log(f"{len(cells)} analysis cells of {cell_size} m")
    if len(cells) > 6000:
        log("WARNING: >6000 cells - this will be slow; consider a larger --cell (e.g. 400) or a smaller area")

    node_ids = list(G.nodes)
    nid = {n: i for i, n in enumerate(node_ids)}
    xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids])
    ntree = cKDTree(xy)

    # ---- edges as arrays -----------------------------------------------------------------
    E = []
    for u, v, d in G.edges(data=True):
        E.append(((xy[nid[u]] + xy[nid[v]]) / 2, d["length"], d.get("highway", "residential"), d.get("speed", 30),
                  d.get("lanes", 1), bool(d.get("sidewalk", False))))
    emid = np.array([e[0] for e in E])
    elen = np.array([e[1] for e in E])
    ehw = [e[2] for e in E]
    espd = np.array([e[3] for e in E], float)
    elan = np.array([e[4] for e in E], float)
    eside = np.array([e[5] for e in E], bool)
    eart = np.array([h in ARTERIAL for h in ehw])
    enonveh = np.array([h in NONVEHICLE for h in ehw])
    etree = cKDTree(emid)
    ewidth = np.where(enonveh, 3.0, np.maximum(3.0, elan * 3.3) + np.where(eside, 3.0, 0))

    # ---- land cover: canopy, buildings, imperviousness, remote sensing ---------------------
    cells["bldg_share"] = area_share(cells, city.buildings)
    cells["canopy_share"] = area_share(cells, city.canopy)
    cells["road_share"] = 0.0
    idx_e = cKDTree(cent).query(emid)[1]
    rs = np.bincount(idx_e, weights=elen * ewidth, minlength=len(cells)) / cell_area
    cells["road_share"] = np.clip(rs, 0, 1)
    cells["imperv"] = np.clip(cells.bldg_share + cells.road_share, 0, 1)

    have_ndvi = city.ndvi_tif and os.path.exists(city.ndvi_tif)
    have_lst = city.lst_tif and os.path.exists(city.lst_tif)
    cells["ndvi"] = zonal_mean(cells, city.ndvi_tif) if have_ndvi else np.nan
    cells["lst_c"] = zonal_mean(cells, city.lst_tif) if have_lst else np.nan
    log(f"remote sensing: NDVI={'yes' if have_ndvi else 'no (proxy)'}  LST={'yes' if have_lst else 'no (proxy)'}")
    veg = np.where(cells.ndvi.notna(), scale(cells.ndvi, .1, .7), np.clip(cells.canopy_share * 1.6, 0, 1))
    cells["veg_frac"] = veg

    # ---- population & socio-economics --------------------------------------------------------
    if city.socio is not None and len(city.socio):
        pop, dep, eld = apportion_socio(cells, city.socio)
        cells["pop"] = pop
        cells["deprivation"] = pd.Series(dep).fillna(np.nanmedian(dep) if np.isfinite(dep).any() else .5).values
        cells["elderly_share"] = pd.Series(eld).fillna(np.nanmedian(eld) if np.isfinite(eld).any() else .13).values
        city.provenance["socio"] = "supplied zones (area-weighted interpolation)"
    else:
        if city.pop_tif and os.path.exists(city.pop_tif):
            cells["pop"] = zonal_sum(cells, city.pop_tif)
            src_pop = "WorldPop-style raster"
        else:
            cells["pop"] = cells.bldg_share * cell_area * 3 / 40.0  # ~3 floors, 40 m2/person: crude proxy
            src_pop = "PROXY from building area"
        cells["deprivation"] = .5
        cells["elderly_share"] = .13
        city.provenance["socio"] = (f"Population: {src_pop}. Deprivation/age: PROXY constants - "
                                    "equity results are NOT meaningful without --socio data.")
        log("WARNING: no deprivation data -> equity module runs on constants")
    cells["pop"] = cells["pop"].clip(lower=0)

    # ---- node mapping -------------------------------------------------------------------------
    d0, i0 = ntree.query(cent)
    cell_node = [node_ids[i] for i in i0]
    cell_snap = d0

    def poi_nodes(gdf):
        if len(gdf) == 0:
            return [], np.array([])
        p = np.c_[gdf.geometry.x, gdf.geometry.y]
        d, i = ntree.query(p)
        return [node_ids[k] for k in i], d

    pn, pd_ = poi_nodes(city.pois)
    poi_by_node: dict = {}
    for n_, c_, d_ in zip(pn, city.pois["category"].values if len(city.pois) else [], pd_):
        if d_ < 150:
            poi_by_node.setdefault(n_, []).append(c_)
    sn, sd = poi_nodes(city.stops)
    stop_by_node: dict = {}
    for n_, m_, t_, d_ in zip(sn, city.stops["mode"].values if len(city.stops) else [],
                              city.stops["trips_per_hr"].values if len(city.stops) else [], sd):
        if d_ < 150:
            stop_by_node.setdefault(n_, []).append((m_, float(t_), float(d_)))

    # ---- green-space network distances ------------------------------------------------------
    parks = city.parks[city.parks.area >= 5000].reset_index(drop=True)
    node_gdf = gpd.GeoDataFrame(geometry=[Point(p) for p in xy], crs=city.crs)
    park_dist: list[dict] = []
    for g in parks.geometry:
        hit = node_gdf.sindex.query(g.buffer(40), predicate="intersects")
        if len(hit) == 0:
            hit = [ntree.query([g.centroid.x, g.centroid.y])[1]]
        park_dist.append(nx.multi_source_dijkstra_path_length(G, [node_ids[h] for h in hit], cutoff=1800, weight="length"))
    park_area = parks.area.values

    # ---- per-cell network accessibility -------------------------------------------------------
    rows = []
    for k in range(len(cells)):
        n0, snap = cell_node[k], cell_snap[k]
        reach = nx.single_source_dijkstra_path_length(G, n0, cutoff=1000, weight="length")
        cats, pts, stops_here = Counter(), [], []
        for m, d in reach.items():
            dd = d + snap
            if dd <= 800:
                pts.append(xy[nid[m]])
                for c in poi_by_node.get(m, ()):
                    cats[c] += 1
            for (mode, trips, sdist) in stop_by_node.get(m, ()):
                stops_here.append((dd + sdist, mode, trips))
        if len(pts) >= 3:
            ped_ratio = MultiPoint(pts).convex_hull.area / (np.pi * 800 ** 2)
        else:
            ped_ratio = 0.0
        # PTAL-inspired accessibility index
        edf = []
        catch = {"bus": 640, "tram": 800, "rail": 960}
        for dist, mode, trips in stops_here:
            if dist <= catch[mode]:
                wt = dist / 80.0
                swt = 0.5 * (60.0 / max(trips, .5)) + (2.0 if mode == "bus" else 0.75)
                edf.append(30.0 / (wt + swt))
        edf.sort(reverse=True)
        ai = (edf[0] + .5 * sum(edf[1:8])) if edf else 0.0
        nearest_stop = min([s[0] for s in stops_here], default=np.nan)
        modes = len({s[1] for s in stops_here if s[0] <= catch[s[1]]})
        # parks
        pdists = [pdict.get(n0, np.inf) + snap for pdict in park_dist]
        dpark = min(pdists, default=np.inf)
        acc_area = float(sum(a for a, dd in zip(park_area, pdists) if dd <= 800))
        if len(parks):
            eu = float(parks.geometry.distance(Point(cent[k])).min())
        else:
            eu = np.inf
        rows.append(dict(pedshed=ped_ratio, dest_cats=len(cats), dest_count=sum(cats.values()), ptal_ai=ai,
                         nearest_stop_m=nearest_stop, transit_modes=modes, park_dist_m=dpark,
                         park_euclid_m=eu, park_area_800=acc_area))
    cells = pd.concat([cells, pd.DataFrame(rows, index=cells.index)], axis=1)
    cells["park_dist_m"] = cells.park_dist_m.replace(np.inf, 3000).clip(upper=3000)
    cells["park_euclid_m"] = cells.park_euclid_m.replace(np.inf, 3000).clip(upper=3000)
    cells["park_detour"] = (cells.park_dist_m / cells.park_euclid_m.clip(lower=60)).clip(lower=1, upper=6)

    # intersection density in 500 m neighbourhood
    jn_idx = [i for i, n_ in enumerate(node_ids) if G.degree[n_] >= 3]
    jtree = cKDTree(xy[jn_idx])
    cnt = np.array([len(x) for x in jtree.query_ball_point(cent, 500)])
    cells["int_density"] = cnt / (np.pi * 0.5 ** 2)

    # ---- junction safety ----------------------------------------------------------------------
    crs_tree = cKDTree(np.c_[city.crossings.geometry.x, city.crossings.geometry.y]) if len(city.crossings) else None
    poi_xy = np.c_[city.pois.geometry.x, city.pois.geometry.y] if len(city.pois) else np.zeros((0, 2))
    ptree = cKDTree(poi_xy) if len(poi_xy) else None
    stop_xy = np.c_[city.stops.geometry.x, city.stops.geometry.y] if len(city.stops) else np.zeros((0, 2))
    str_ = cKDTree(stop_xy) if len(stop_xy) else None
    J = []
    for i in jn_idx:
        n_ = node_ids[i]
        inc = [G.edges[n_, m] for m in G[n_]]
        veh = [e for e in inc if e.get("highway") not in NONVEHICLE]
        if not veh:
            continue
        vmax = max(e["speed"] for e in veh)
        lanes = max(e["lanes"] for e in veh)
        legs = len(veh)
        if legs < 3:
            continue
        kind = None
        if crs_tree is not None:
            for j in crs_tree.query_ball_point(xy[i], 30):
                kd = city.crossings["kind"].iloc[j]
                kind = "signal" if (kd == "signal" or kind == "signal") else "marked"
        any_side = any(e.get("sidewalk") for e in veh)
        sev = float(ped_injury_risk(vmax))
        widthf = .6 + .4 * min(1, lanes * 3.3 / 13)
        mit = {"signal": .6, "marked": .8}.get(kind, 1.0) * (1.0 if any_side else 1.2)
        risk = float(min(1, sev * widthf * (1 + .12 * (legs - 3)) * mit / .9) * 100)
        dem = 0.0
        if ptree is not None:
            dem += min(1, len(ptree.query_ball_point(xy[i], 200)) / 15)
        if str_ is not None:
            dem += .5 * min(1, len(str_.query_ball_point(xy[i], 120)) / 2)
        J.append(dict(node=n_, x=xy[i][0], y=xy[i][1], vmax_kmh=vmax, lanes=lanes, legs=legs,
                      crossing=kind or "none", sidewalk=any_side, risk=risk, demand=min(1, dem)))
    junc = pd.DataFrame(J)
    junc = gpd.GeoDataFrame(junc, geometry=gpd.points_from_xy(junc.x, junc.y), crs=city.crs)
    junc["priority"] = junc.risk * (.5 + .5 * junc.demand)
    junc["unprotected"] = (junc.crossing == "none") & (junc.vmax_kmh >= 40)
    jt = cKDTree(np.c_[junc.x, junc.y])
    jr, jp, jmax = [], [], []
    for k in range(len(cells)):
        ids = jt.query_ball_point(cent[k], 300)
        if not ids:
            jr.append(np.nan); jmax.append(np.nan); jp.append(np.nan)
            continue
        d = np.hypot(*(np.c_[junc.x.values[ids], junc.y.values[ids]] - cent[k]).T)
        w = 1 / (1 + d / 150)
        jr.append(np.average(junc.risk.values[ids], weights=w))
        jmax.append(junc.risk.values[ids].max())
        jp.append(junc.unprotected.values[ids].mean())
    cells["junc_risk"] = jr
    cells["junc_risk_max"] = jmax
    cells["junc_unprot_share"] = jp

    # ---- pedestrian comfort -------------------------------------------------------------------
    sw, low, art = [], [], []
    for k, ids in enumerate(etree.query_ball_point(cent, 300)):
        if not ids:
            sw.append(np.nan); low.append(np.nan); art.append(np.nan)
            continue
        w = elen[ids]
        sw.append(np.average(eside[ids], weights=w))
        low.append(np.average(espd[ids] <= 30, weights=w))
        art.append(np.average(eart[ids], weights=w))
    cells["sidewalk_share"] = sw
    cells["lowspeed_share"] = low
    cells["arterial_share"] = art

    # ====================== SCORES (0-100, higher = better) =====================================
    c = cells
    conn = scale(c.int_density, 30, 140)
    ped = scale(c.pedshed, .2, .55)
    dest = .6 * (c.dest_cats / len(CATS)) + .4 * scale(c.dest_count, 0, 25)
    c["walk"] = 100 * (.3 * conn + .3 * ped + .4 * dest)

    jr_ = c.junc_risk.fillna(0)
    jm_ = c.junc_risk_max.fillna(0)
    c["safety"] = 100 - (.6 * jr_ + .4 * jm_)

    shade = scale(c.veg_frac, 0, .5)
    c["comfort"] = 100 * (.35 * c.sidewalk_share.fillna(0) + .25 * c.lowspeed_share.fillna(0)
                          + .20 * shade + .20 * (1 - c.arterial_share.fillna(.5)))

    c["transit"] = 100 * np.clip(c.ptal_ai / 25, 0, 1) * .7 + 100 * (.15 * scale(c.nearest_stop_m.fillna(1000), 800, 100)
                                                                        + .15 * np.clip(c.transit_modes / 2, 0, 1))

    access = scale(c.park_dist_m, 1200, 300)
    canopy30 = np.clip(c.veg_frac / .5, 0, 1)
    pop800 = np.array([c["pop"].values[ids].sum() for ids in cKDTree(cent).query_ball_point(cent, 800)])
    c["park_m2_pp"] = c.park_area_800 / np.maximum(pop800, 1)
    c["green"] = 100 * (.5 * access + .3 * canopy30 + .2 * scale(c.park_m2_pp, 0, 30))

    # heat: IPCC-style exposure / sensitivity / adaptive capacity
    lst_pct = c.lst_c.rank(pct=True) if c.lst_c.notna().any() else None
    exp_proxy = .5 * c.imperv + .5 * (1 - c.veg_frac)
    c["heat_exposure"] = 100 * (.5 * lst_pct.fillna(.5) + .5 * exp_proxy if lst_pct is not None else exp_proxy)
    pop_dens = c["pop"] / (cell_area / 1e6)
    c["heat_sensitivity"] = 100 * (.4 * c.deprivation + .4 * scale(c.elderly_share, .05, .25) + .2 * scale(pop_dens, 1000, 12000))
    c["heat_adaptive"] = 100 * (.7 * access + .3 * (1 - c.deprivation))
    c["heat_vulnerability"] = (c.heat_exposure + c.heat_sensitivity + (100 - c.heat_adaptive)) / 3
    c["heat"] = 100 - c.heat_vulnerability

    c["overall"] = sum(weights[s] * c[s] for s in SCORES) / sum(weights.values())

    # ---- equity ----------------------------------------------------------------------------------
    c["need"] = .7 * c.deprivation + .3 * scale(c.elderly_share, .05, .25)
    c["deficit"] = 1 - c.overall / 100
    c["priority"] = c.need * c.deficit
    eq = equity_table(c)

    # ---- AI layer 1: typologies ----------------------------------------------------------------------
    typologies = cluster_typologies(c)
    # ---- AI layer 2: recommendation engine -------------------------------------------------------------
    zones, recs = recommend(c, junc, cell_size)

    s = summarise(city, c, junc, eq, typologies, zones, recs)
    return Result(city, c, junc, zones, recs, eq, typologies, s, weights)


# ----------------------------------------------------------------------------------------
# Equity
# ----------------------------------------------------------------------------------------
def _wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def concentration_index(h, rank_var, w):
    """Population-weighted concentration index of h ranked by rank_var (ascending = most deprived first
    when rank_var = -deprivation). CI > 0: benefit concentrated among the LEAST deprived."""
    h, rv, w = [np.asarray(a, float) for a in (h, rank_var, w)]
    o = np.argsort(rv)
    h, w = h[o], w[o]
    cw = np.cumsum(w) / w.sum()
    r = cw - w / w.sum() / 2
    mu = np.average(h, weights=w)
    cov = np.average((h - mu) * (r - np.average(r, weights=w)), weights=w)
    return float(2 * cov / mu) if mu else np.nan


def equity_table(c):
    w = c["pop"].values
    ok = w > 0
    d = c.deprivation.values
    q = pd.qcut(pd.Series(d).rank(method="first"), 5, labels=False).values  # 0 = least deprived
    rows = []
    for s in SCORES + ["overall"]:
        h = c[s].values
        qm = [_wmean(h[(q == k) & ok], w[(q == k) & ok]) for k in range(5)]
        rows.append(dict(indicator=s, city_mean=_wmean(h, w), least_deprived_q=qm[0], most_deprived_q=qm[4],
                         gap=qm[0] - qm[4], concentration_index=concentration_index(h[ok], -d[ok], w[ok])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# AI layer 1 - unsupervised typology discovery
# ----------------------------------------------------------------------------------------
def cluster_typologies(c, k=5, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    feats = SCORES
    pretty = {"walk": "walkability", "safety": "junction safety", "comfort": "comfort", "transit": "transit",
              "green": "green access", "heat": "heat resilience"}
    X = StandardScaler().fit_transform(c[feats].fillna(c[feats].median()))
    k = min(k, max(2, len(c) // 8))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    c["typology_id"] = km.labels_
    labels = {}
    for i, cen in enumerate(km.cluster_centers_):
        order = np.argsort(-np.abs(cen))[:2]
        bits = [("strong " if cen[j] > 0 else "weak ") + pretty[feats[j]] for j in order if abs(cen[j]) > .35]
        labels[i] = (", ".join(bits) if bits else "average / mixed")
    c["typology"] = c.typology_id.map(labels)
    # make labels unique
    seen = Counter()
    for i in list(labels):
        seen[labels[i]] += 1
        if seen[labels[i]] > 1:
            labels[i] = f"{labels[i]} ({seen[labels[i]]})"
    c["typology"] = c.typology_id.map(labels)
    return labels


# ----------------------------------------------------------------------------------------
# AI layer 2 - evidence-linked recommendation engine
# ----------------------------------------------------------------------------------------
@dataclass
class Intervention:
    id: str
    title: str
    trigger: callable        # zone metrics dict -> severity 0..1
    action: callable         # zone metrics dict, junction text -> str
    evidence: str
    cost: str
    horizon: str
    targets: str


def _top_junctions(junc, geom, n=3):
    j = junc[junc.geometry.within(geom.buffer(150))].sort_values("priority", ascending=False).head(n)
    return j


INTERVENTIONS = [
    Intervention("speed", "Speed management & safer crossings at high-risk junctions",
                 lambda z: float(scale(z["junc_risk"], 22, 55)),
                 lambda z, jt: f"Introduce 30 km/h limits and upgrade crossings (raised tables, tightened corner radii, "
                               f"continuous footways) starting with the worst junctions{jt}.",
                 "Pedestrian injury severity rises steeply with impact speed; lower speeds and shorter, "
                 "protected crossings are core Safe System measures with strong international evidence.",
                 "€€", "0-2 yrs", "safety, comfort"),
    Intervention("crossing_gap", "Close crossing gaps on fast roads",
                 lambda z: float(scale(z["junc_unprot_share"], .25, .8)),
                 lambda z, jt: f"Add marked/signalised crossings at junctions on ≥40 km/h roads that have none "
                               f"({z['junc_unprot_share']*100:.0f}% of local junctions){jt}.",
                 "Provision of formal crossings where speeds and flows are high reduces conflict exposure; "
                 "prioritise where destination/transit demand is high.",
                 "€€", "0-2 yrs", "safety, walk"),
    Intervention("trees", "Street trees, shade corridors and cool surfaces",
                 lambda z: float(np.clip((.30 - z["veg_frac"] * .5) / .30, 0, 1) * scale(z["heat_exposure"], 35, 70)),
                 lambda z, jt: "Plant street trees on the main walking routes (prioritise streets leading to schools, stops and "
                               "local centres), de-pave where feasible and use higher-albedo materials.",
                 "Canopy shading and evapotranspiration lower surface and perceived temperatures; effect size is "
                 "context dependent, so monitor with LST.",
                 "€€", "2-5 yrs", "heat, comfort, green"),
    Intervention("park_new", "Create new green space / pocket parks",
                 lambda z: float(scale(z["park_dist_m"], 300, 1000)) * (1.0 if z["park_detour"] < 1.5 else .6),
                 lambda z, jt: f"Residents are on average {z['park_dist_m']:.0f} m from public green space (3-30-300 target: 300 m). "
                               "Convert vacant plots, parking or road space into pocket parks/parklets.",
                 "Proximity to green space is associated with physical activity and wellbeing; the 300 m "
                 "guideline is widely used (3-30-300 rule, WHO/EU guidance).",
                 "€€€", "2-5 yrs", "green, heat"),
    Intervention("park_access", "Open up access to existing parks",
                 lambda z: float(scale(z["park_detour"], 1.4, 2.4)) * float(z["park_euclid_m"] < 500),
                 lambda z, jt: f"A park exists ~{z['park_euclid_m']:.0f} m away as the crow flies but needs "
                               f"{z['park_dist_m']:.0f} m on foot (detour ×{z['park_detour']:.1f}). Add entrances / "
                               "pedestrian links and fix severance (barriers, busy roads).",
                 "Network detour, not straight-line distance, drives real access; new entrances are often the "
                 "cheapest way to raise coverage.",
                 "€", "0-2 yrs", "green, walk"),
    Intervention("permeability", "Pedestrian links through low-permeability blocks",
                 lambda z: float(scale(z["pedshed"], .45, .2)),
                 lambda z, jt: f"Pedshed ratio is only {z['pedshed']:.2f}. Add footpath/cycle cut-throughs between "
                               "cul-de-sacs and across superblocks.",
                 "Street connectivity and direct routes are consistently linked to more walking trips.",
                 "€€", "1-3 yrs", "walk"),
    Intervention("transit", "Improve public transport coverage and frequency",
                 lambda z: float(scale(z["transit"], 55, 15)),
                 lambda z, jt: (f"Nearest stop is ~{z['nearest_stop_m']:.0f} m away: add stops / on-demand feeder service."
                                if z["nearest_stop_m"] > 600 else
                                f"Coverage exists but accessibility index is low ({z['ptal_ai']:.1f}): increase frequency on "
                                "existing routes and add interchange links."),
                 "Service frequency and walk-access time jointly determine usable connectivity (PTAL-style logic).",
                 "€€€", "1-3 yrs", "transit"),
    Intervention("destinations", "Enable local everyday destinations",
                 lambda z: float(scale(z["dest_cats"], 4, 1)),
                 lambda z, jt: f"Only {z['dest_cats']:.1f} of 6 everyday service types are within a 10-minute walk. "
                               "Use planning policy to enable ground-floor retail, health and community uses.",
                 "Land-use mix within walking distance underpins the '15-minute neighbourhood' concept.",
                 "€", "2-5 yrs", "walk"),
    Intervention("sidewalks", "Complete and widen the footway network",
                 lambda z: float(scale(z["sidewalk_share"], .9, .5)),
                 lambda z, jt: f"Only {z['sidewalk_share']*100:.0f}% of nearby street length has footway provision. "
                               "Fill sidewalk gaps along routes to stops, schools and shops first.",
                 "Missing or discontinuous footways force pedestrians into the carriageway, harming safety and accessibility.",
                 "€€", "1-3 yrs", "comfort, safety"),
    Intervention("heat_refuge", "Heat-health measures: cool refuges, shaded stops, drinking water",
                 lambda z: float(scale(z["heat_vulnerability"], 45, 65)),
                 lambda z, jt: "Designate cool refuges (libraries, shaded parks), shade bus stops, add drinking fountains, "
                               "and target heat-warning outreach to older residents.",
                 "Heat risk is the product of exposure, sensitivity and limited adaptive capacity; "
                 "protecting sensitive groups is the fastest lever.",
                 "€", "0-1 yrs", "heat"),
]


def recommend(c, junc, cell_size, top_q=.8):
    # flag priority cells
    thr = c.priority.quantile(top_q)
    c["is_priority"] = (c.priority >= thr) & (c["pop"] > c["pop"].quantile(.2))
    pcells = c[c.is_priority]
    if len(pcells) == 0:
        return [], pd.DataFrame()
    pc = np.c_[pcells.geometry.centroid.x, pcells.geometry.centroid.y]
    tree = cKDTree(pc)
    g = nx.Graph()
    g.add_nodes_from(range(len(pcells)))
    for a, b in tree.query_pairs(cell_size * 1.5):
        g.add_edge(a, b)
    comps = sorted(nx.connected_components(g), key=lambda s: -(pcells.priority.values[list(s)] * (pcells['pop'].values[list(s)] + 1)).sum())
    zones, rows = [], []
    num_cols = ["junc_risk", "junc_unprot_share", "veg_frac", "heat_exposure", "heat_vulnerability", "park_dist_m",
                "park_detour", "park_euclid_m", "pedshed", "transit", "nearest_stop_m", "ptal_ai", "dest_cats",
                "sidewalk_share", "walk", "safety", "comfort", "green", "heat", "overall", "deprivation", "need"]
    for zi, comp in enumerate(comps):
        sub = pcells.iloc[sorted(comp)]
        if len(sub) < 2 and zi >= 6:  # ignore isolated singletons beyond the top few
            continue
        name = f"Zone {chr(65 + len(zones))}" if len(zones) < 26 else f"Zone {len(zones)+1}"
        w = sub["pop"].values + 1
        z = {k: _wmean(sub[k].values, w) for k in num_cols}
        geom = shapely.union_all(sub.geometry.values)
        jx = _top_junctions(junc, geom)
        jt = ""
        if len(jx):
            jt = " [junction IDs: " + ", ".join(f"#{int(n)} (risk {r:.0f}, {v:.0f} km/h)" for n, r, v in
                                                zip(jx.node, jx.risk, jx.vmax_kmh)) + "]"
        zinfo = dict(name=name, cells=len(sub), pop=float(sub["pop"].sum()), geometry=geom, metrics=z,
                     priority=float(sub.priority.mean()), junctions=jx)
        sev_list = []
        for iv in INTERVENTIONS:
            try:
                sv = float(iv.trigger(z))
            except Exception:
                sv = 0.0
            if sv > .25:
                sev_list.append((sv, iv))
        sev_list.sort(key=lambda t: -t[0])
        zinfo["recs"] = []
        for rank, (sv, iv) in enumerate(sev_list[:5], 1):
            r = dict(zone=name, rank=rank, intervention=iv.title, severity=round(sv, 2), action=iv.action(z, jt),
                     evidence=iv.evidence, cost=iv.cost, horizon=iv.horizon, indicators=iv.targets,
                     population=round(zinfo["pop"]),
                     confidence="high" if sv > .6 else "medium")
            rows.append(r)
            zinfo["recs"].append(r)
        zones.append(zinfo)
        if len(zones) >= 8:
            break
    return zones, pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Summary / narrative
# ----------------------------------------------------------------------------------------
def summarise(city, c, junc, eq, typ, zones, recs):
    w = c["pop"].values
    s = dict(city=city.name, cells=len(c), population=float(c["pop"].sum()),
             provenance=city.provenance,
             scores={k: _wmean(c[k].values, w) for k in SCORES + ["overall"]},
             pct_pop_park300=float(100 * _wmean((c.park_dist_m <= 300).astype(float), w)),
             pct_pop_transit_ptal_ge_5=float(100 * _wmean((c.ptal_ai >= 5).astype(float), w)),
             pct_pop_canopy_ge30=float(100 * _wmean((c.veg_frac >= .5).astype(float), w)),
             junctions=len(junc), junctions_high_risk=int((junc.risk >= 45).sum()),
             junctions_unprotected_fast=int(junc.unprotected.sum()),
             equity=eq.round(3).to_dict(orient="records"),
             zones=[dict(name=z["name"], population=round(z["pop"]), cells=z["cells"],
                         metrics={k: round(float(v), 2) for k, v in z["metrics"].items() if np.isfinite(v)},
                         recommendations=[dict(title=r["intervention"], action=r["action"], severity=r["severity"])
                                          for r in z["recs"]]) for z in zones],
             typologies=typ)
    return s


def template_brief(s):
    sc = s["scores"]
    worst = min(SCORES, key=lambda k: sc[k])
    best = max(SCORES, key=lambda k: sc[k])
    ov = [e for e in s["equity"] if e["indicator"] == "overall"][0]
    lines = [f"Across {s['cells']} cells ({s['population']:.0f} residents modelled), the overall score is "
             f"{sc['overall']:.0f}/100. Strongest dimension: {SCORE_LABELS[best].lower()} ({sc[best]:.0f}); "
             f"weakest: {SCORE_LABELS[worst].lower()} ({sc[worst]:.0f}).",
             f"{s['pct_pop_park300']:.0f}% of residents live within 300 m (network) of public green space and "
             f"{s['pct_pop_canopy_ge30']:.0f}% live in cells that meet a ~30% vegetation benchmark. "
             f"{s['junctions_high_risk']} of {s['junctions']} junctions screen as high pedestrian risk; "
             f"{s['junctions_unprotected_fast']} are on ≥40 km/h roads without any crossing."]
    if "PROXY" not in s["provenance"].get("socio", ""):
        direction = "favour less-deprived" if ov["concentration_index"] > 0 else "favour more-deprived"
        lines.append(f"Equity: the most-deprived quintile scores {ov['most_deprived_q']:.0f} vs {ov['least_deprived_q']:.0f} "
                     f"for the least-deprived (gap {ov['gap']:.0f} pts; concentration index {ov['concentration_index']:+.3f}, i.e. "
                     f"outcomes {direction} neighbourhoods).")
    if s["zones"]:
        z = s["zones"][0]
        top = z["recommendations"][0]["title"] if z["recommendations"] else "n/a"
        lines.append(f"Highest-priority area: {z['name']} ({z['population']} residents). Lead action: {top}.")
    return "\n\n".join(lines)


def llm_brief(s):
    """Optional: grounded narrative via Claude. Falls back to the template if no key / SDK."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return template_brief(s), "template (no ANTHROPIC_API_KEY)"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        payload = json.dumps({k: s[k] for k in ("city", "scores", "pct_pop_park300", "pct_pop_transit_ptal_ge_5",
                                                  "pct_pop_canopy_ge30", "junctions", "junctions_high_risk",
                                                  "junctions_unprotected_fast", "equity", "zones", "typologies",
                                                  "provenance")}, default=float)
        msg = client.messages.create(
            model=os.environ.get("URBANLENS_MODEL", "claude-sonnet-5-5"), max_tokens=900,
            system=("You are an urban planning analyst. Write a concise (max 350 words) planning brief for city officials "
                    "using ONLY the numbers in the JSON. Never invent figures. State clearly if data is synthetic or proxy-based. "
                    "Structure: headline finding, equity, top 3 priority actions with why, caveats."),
            messages=[{"role": "user", "content": payload}])
        return "".join(b.text for b in msg.content if b.type == "text"), "Claude"
    except Exception as e:  # pragma: no cover
        return template_brief(s), f"template (LLM unavailable: {e})"


# ----------------------------------------------------------------------------------------
# Outputs: interactive map, HTML report, data exports
# ----------------------------------------------------------------------------------------
def build_map(res: Result, path: str):
    import branca.colormap as cm
    import folium
    from folium import plugins

    c = res.cells.to_crs(4326)
    junc = res.junctions.to_crs(4326)
    ctr = c.geometry.union_all().centroid if hasattr(c.geometry, "union_all") else c.unary_union.centroid
    m = folium.Map(location=[ctr.y, ctr.x], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM basemap", show=False).add_to(m)
    cmap = cm.LinearColormap(["#b2182b", "#ef8a62", "#fddbc7", "#d9f0d3", "#7fbf7b", "#1b7837"], vmin=0, vmax=100,
                             caption="Score (0 = poor, 100 = good)")
    fields = [("overall", "Overall"), ("walk", "Walkability"), ("safety", "Junction safety"), ("comfort", "Pedestrian comfort"),
              ("transit", "Transit connectivity"), ("green", "Green access"), ("heat", "Heat resilience"),
              ("heat_vulnerability", "Climate vulnerability (idx)"), ("need", "Social need"), ("priority", "Priority index")]
    popup_cols = [("pop", "Population", "{:.0f}"), ("overall", "Overall", "{:.0f}"), ("walk", "Walk", "{:.0f}"),
                  ("safety", "Safety", "{:.0f}"), ("comfort", "Comfort", "{:.0f}"), ("transit", "Transit", "{:.0f}"),
                  ("green", "Green", "{:.0f}"), ("heat", "Heat resilience", "{:.0f}"), ("lst_c", "LST °C", "{:.1f}"),
                  ("ndvi", "NDVI", "{:.2f}"), ("park_dist_m", "Park dist (m)", "{:.0f}"),
                  ("deprivation", "Deprivation", "{:.2f}"), ("typology", "Typology", "{}")]

    def popup(r):
        rows = "".join(f"<tr><td>{lab}</td><td><b>{fmt.format(r[k])}</b></td></tr>" for k, lab, fmt in popup_cols
                       if k in r and pd.notna(r[k]))
        return f"<table style='font:12px sans-serif'>{rows}</table>"

    c["geometry"] = shapely.set_precision(c.geometry.values, 1e-5)
    c["_popup"] = c.apply(popup, axis=1)
    for key, label in fields:
        grp = folium.FeatureGroup(name=f"{label}", show=(key == "overall"))
        if key in ("heat_vulnerability", "need", "priority"):
            vmin, vmax = float(c[key].min()), float(c[key].max())
            cmx = cm.LinearColormap(["#ffffcc", "#fd8d3c", "#800026"], vmin=vmin, vmax=vmax)
            colorf = (lambda v, cmx=cmx: cmx(v))
        else:
            colorf = (lambda v: cmap(v))
        gj = json.loads(c[[key, "geometry"]].round(2).to_json(drop_id=True))
        folium.GeoJson(gj, style_function=lambda f, key=key, colorf=colorf: {
            "fillColor": colorf(f["properties"][key]) if f["properties"][key] is not None else "#999",
            "color": "#ffffff", "weight": .4, "fillOpacity": .68}).add_to(grp)
        grp.add_to(m)
    # typologies
    pal = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948"]
    grp = folium.FeatureGroup(name="AI typologies", show=False)
    gj_t = json.loads(c[["typology_id", "geometry"]].to_json(drop_id=True))
    folium.GeoJson(gj_t, style_function=lambda f: {"fillColor": pal[int(f["properties"]["typology_id"]) % 7], "color": "#fff",
                                                  "weight": .4, "fillOpacity": .65},
                   ).add_to(grp)
    grp.add_to(m)
    # click layer: invisible polygons on top of whichever colour layer is active (keeps file small)
    grp = folium.FeatureGroup(name="Cell details (click)", show=True)
    gj_d = json.loads(c[["_popup", "typology", "overall", "geometry"]].round(1).to_json(drop_id=True))
    folium.GeoJson(gj_d, style_function=lambda f: {"fillColor": "#000", "fillOpacity": 0, "weight": 0},
                   tooltip=folium.GeoJsonTooltip(fields=["overall", "typology"], aliases=["Overall:", "Typology:"],
                                                 style="font-size:12px"),
                   popup=folium.GeoJsonPopup(fields=["_popup"], labels=False, max_width=260)).add_to(grp)
    grp.add_to(m)
    # junctions
    grp = folium.FeatureGroup(name="Junction risk", show=False)
    jcm = cm.LinearColormap(["#1a9850", "#fee08b", "#d73027"], vmin=0, vmax=80)
    for _, r in junc.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3 + r.risk / 25, color=jcm(r.risk), weight=1, fill=True,
                            fill_opacity=.85,
                            tooltip=(f"Junction #{int(r.node)}: risk {r.risk:.0f}/100 · {r.vmax_kmh:.0f} km/h · "
                                     f"{int(r.lanes)} lanes · crossing: {r.crossing}")).add_to(grp)
    grp.add_to(m)
    # parks & stops
    g = folium.FeatureGroup(name="Parks", show=False)
    folium.GeoJson(res.city.parks.to_crs(4326)[["geometry"]].to_json(),
                   style_function=lambda f: {"fillColor": "#2e7d32", "color": "#1b5e20", "weight": 1, "fillOpacity": .5}).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup(name="Transit stops", show=False)
    st = res.city.stops.to_crs(4326)
    col = {"bus": "#1f77b4", "tram": "#9467bd", "rail": "#d62728"}
    for _, r in st.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, color=col[r["mode"]], fill=True, fill_opacity=.9,
                            tooltip=f"{r['mode']} · {r.trips_per_hr:.0f} trips/hr").add_to(g)
    g.add_to(m)
    # priority zones with recommendations
    g = folium.FeatureGroup(name="Priority zones + recommendations", show=True)
    for z in res.zones:
        geom = gpd.GeoSeries([z["geometry"]], crs=res.city.crs).to_crs(4326).iloc[0]
        lis = "".join(f"<li><b>{html.escape(r['intervention'])}</b> <i>({r['cost']}, {r['horizon']})</i><br>"
                      f"{html.escape(r['action'])}</li>" for r in z["recs"])
        pop = (f"<div style='font:12px sans-serif;max-width:420px'><h4 style='margin:0'>{z['name']}</h4>"
               f"{z['pop']:.0f} residents · overall score {z['metrics']['overall']:.0f}<ol style='padding-left:16px'>{lis}</ol></div>")
        folium.GeoJson(shapely.geometry.mapping(geom), style_function=lambda f: {"fillColor": "none", "color": "#111",
                                                                                 "weight": 3, "dashArray": "6 4"},
                       tooltip=z["name"], popup=folium.Popup(pop, max_width=450)).add_to(g)
    g.add_to(m)
    cmap.add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True, zoom_level_offset=-5).add_to(m)
    s = res.summary
    panel = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.93);padding:10px 12px;"
             f"border-radius:8px;font:12px sans-serif;box-shadow:0 1px 6px rgba(0,0,0,.3);max-width:260px;color:#111'>"
             f"<b>{html.escape(s['city'])}</b><br>Overall: <b>{s['scores']['overall']:.0f}</b>/100<br>"
             + "".join(f"{SCORE_LABELS[k].split(' &')[0]}: {s['scores'][k]:.0f}<br>" for k in SCORES)
             + f"<span style='color:#a00'>{'SYNTHETIC DEMO DATA' if s['provenance'].get('synthetic') else ''}</span></div>")
    m.get_root().html.add_child(folium.Element(panel))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(path)


def build_report(res: Result, brief: str, brief_src: str, path: str):
    s = res.summary
    esc = html.escape

    def bar(v):
        col = "#1b7837" if v >= 66 else "#e08214" if v >= 40 else "#b2182b"
        return (f"<div class='bar'><div style='width:{max(0, min(100, v)):.0f}%;background:{col}'></div></div>")

    cards = "".join(f"<div class='card'><div class='k'>{esc(SCORE_LABELS[k])}</div><div class='v'>{s['scores'][k]:.0f}</div>"
                    f"{bar(s['scores'][k])}</div>" for k in SCORES)
    eq = "".join(f"<tr><td>{esc(SCORE_LABELS.get(e['indicator'], 'Overall'))}</td><td>{e['least_deprived_q']:.1f}</td>"
                 f"<td>{e['most_deprived_q']:.1f}</td><td>{e['gap']:+.1f}</td><td>{e['concentration_index']:+.3f}</td></tr>"
                 for e in s["equity"])
    zones = ""
    for z in res.zones:
        recs = "".join(f"<li><b>{esc(r['intervention'])}</b> <span class='tag'>{r['cost']} · {r['horizon']} · "
                       f"conf: {r['confidence']}</span><br>{esc(r['action'])}<br><small><i>Evidence:</i> {esc(r['evidence'])}</small></li>"
                       for r in z["recs"])
        m = z["metrics"]
        zones += (f"<section><h3>{z['name']} <small>· {z['pop']:.0f} residents · {z['cells']} cells · overall {m['overall']:.0f}</small></h3>"
                  f"<p class='m'>walk {m['walk']:.0f} · safety {m['safety']:.0f} · comfort {m['comfort']:.0f} · transit {m['transit']:.0f} · "
                  f"green {m['green']:.0f} · heat resilience {m['heat']:.0f} · deprivation {m['deprivation']:.2f}</p><ol>{recs}</ol></section>")
    typ = res.cells.groupby("typology").agg(cells=("cell_id", "count"), pop=("pop", "sum"), overall=("overall", "mean")).reset_index()
    typ_rows = "".join(f"<tr><td>{esc(r.typology)}</td><td>{r.cells}</td><td>{r['pop']:.0f}</td><td>{r.overall:.0f}</td></tr>"
                       for _, r in typ.sort_values("overall").iterrows())
    warn = ""
    if s["provenance"].get("synthetic"):
        warn = "<div class='warn'>⚠ Synthetic demonstration data - results illustrate the workflow, not a real place.</div>"
    elif "PROXY" in s["provenance"].get("socio", ""):
        warn = "<div class='warn'>⚠ No socio-economic data supplied: equity results are not meaningful.</div>"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UrbanLens report - {esc(s['city'])}</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f4f5f7;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#eaeaea;--mut:#9aa;--card:#23262c;--line:#3a3d44}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:960px;margin:0 auto;padding:24px}}
h1{{margin:.2em 0}}h2{{margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:var(--card);padding:12px;border-radius:10px}}.k{{font-size:12px;color:var(--mut)}}.v{{font-size:30px;font-weight:700}}
.bar{{height:6px;background:var(--line);border-radius:3px}}.bar div{{height:6px;border-radius:3px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}}
.tag{{font-size:12px;background:var(--card);padding:1px 6px;border-radius:8px;color:var(--mut)}}
.warn{{background:#fff3cd;color:#664d03;padding:10px;border-radius:8px;margin:12px 0}}
section{{background:var(--card);padding:4px 16px 8px;border-radius:10px;margin:12px 0}}.m{{color:var(--mut);font-size:13px}}
li{{margin-bottom:8px}}small{{color:var(--mut)}}pre{{white-space:pre-wrap;font:inherit}}
</style></head><body>
<h1>UrbanLens report</h1><div>{esc(s['city'])} · {s['cells']} analysis cells · {s['population']:.0f} residents modelled</div>{warn}
<h2>Planning brief <small>({esc(brief_src)})</small></h2><pre>{esc(brief)}</pre>
<h2>Indicator scores (population-weighted, 0-100)</h2><div class="grid"><div class="card"><div class="k">Overall</div>
<div class="v">{s['scores']['overall']:.0f}</div>{bar(s['scores']['overall'])}</div>{cards}</div>
<h2>Key benchmarks</h2><ul><li>{s['pct_pop_park300']:.0f}% of residents within 300 m network distance of public green space</li>
<li>{s['pct_pop_canopy_ge30']:.0f}% of residents in cells meeting the ~30% vegetation benchmark</li>
<li>{s['pct_pop_transit_ptal_ge_5']:.0f}% of residents with PTAL-style accessibility index ≥ 5</li>
<li>{s['junctions_high_risk']} / {s['junctions']} junctions high-risk; {s['junctions_unprotected_fast']} unprotected on ≥40 km/h roads</li></ul>
<h2>Neighbourhood equity</h2><table><tr><th>Indicator</th><th>Least-deprived quintile</th><th>Most-deprived quintile</th><th>Gap</th><th>Concentration index</th></tr>{eq}</table>
<p><small>Concentration index &gt; 0: outcomes concentrated among the least-deprived; &lt; 0: among the most-deprived.</small></p>
<h2>Priority zones &amp; recommendations</h2>{zones}
<h2>AI-discovered neighbourhood typologies</h2><table><tr><th>Typology (k-means on indicators)</th><th>Cells</th><th>Population</th><th>Mean overall</th></tr>{typ_rows}</table>
<h2>Method &amp; limitations</h2><ul>
<li>Indicators use <b>network distance</b> (Dijkstra on the street graph), not straight-line buffers.</li>
<li>Junction risk is a transparent screening heuristic (speed→severity curve × width × legs × mitigation), <b>not</b> a crash-prediction model; calibrate with local crash data.</li>
<li>Transit uses a PTAL-inspired index; real headways need GTFS. Heat uses IPCC exposure/sensitivity/adaptive-capacity logic with Landsat-style NDVI/LST where supplied, otherwise land-cover proxies.</li>
<li>Recommendations are rule-based and traceable to the metrics shown; costs/horizons are indicative classes. Weights: {esc(json.dumps(res.weights))}.</li>
<li>Data provenance: {esc(json.dumps(s['provenance']))}</li></ul></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def run(city: CityData, out: str, cell=250, use_llm=False, weights=None):
    os.makedirs(out, exist_ok=True)
    print(f"[UrbanLens] {city.name}")
    res = analyze(city, cell_size=cell, weights=weights)
    brief, src = llm_brief(res.summary) if use_llm else (template_brief(res.summary), "template")
    cells = res.cells.copy()
    cells.drop(columns=[c for c in cells.columns if c.startswith("_")]).to_crs(4326).to_file(
        os.path.join(out, "cells.geojson"), driver="GeoJSON")
    res.junctions.to_crs(4326).to_file(os.path.join(out, "junctions.geojson"), driver="GeoJSON")
    res.recs.to_csv(os.path.join(out, "recommendations.csv"), index=False)
    res.equity.to_csv(os.path.join(out, "equity.csv"), index=False)
    res.summary["brief"], res.summary["brief_source"] = brief, src
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res.summary, f, indent=2, default=float)
    if res.zones:
        zg = gpd.GeoDataFrame([{"name": z["name"], "pop": z["pop"], "cells": z["cells"],
                                **{k: float(z["metrics"][k]) for k in SCORES + ["overall", "deprivation"]}}
                               for z in res.zones], geometry=[z["geometry"] for z in res.zones], crs=res.city.crs)
        zg.to_crs(4326).to_file(os.path.join(out, "zones.geojson"), driver="GeoJSON")
    build_map(res, os.path.join(out, "map.html"))
    build_report(res, brief, src, os.path.join(out, "report.html"))
    try:
        from dashboard import build_dashboard
        build_dashboard(out)
    except Exception as e:  # dashboard is optional
        print("  · dashboard skipped:", e)
    print(f"[UrbanLens] done -> {out}/ (dashboard.html, map.html, report.html, + data files)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run on the synthetic offline city")
    d.add_argument("--out", default="out_demo"); d.add_argument("--cell", type=int, default=250)
    d.add_argument("--llm", action="store_true")
    o = sub.add_parser("osm", help="run on a real place from OpenStreetMap (needs internet + osmnx)")
    o.add_argument("place"); o.add_argument("--out", default="out_osm"); o.add_argument("--cell", type=int, default=250)
    o.add_argument("--socio"); o.add_argument("--pop"); o.add_argument("--bbox", help="west,south,east,north (fallback if place lookup fails)"); o.add_argument("--ndvi"); o.add_argument("--lst"); o.add_argument("--llm", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        city = synthetic_city(os.path.join(a.out, "rs"))
    else:
        city = load_osm(a.place, a.socio, a.ndvi, a.lst, a.pop, a.bbox)
    run(city, a.out, a.cell, a.llm)


if __name__ == "__main__":
    main()
