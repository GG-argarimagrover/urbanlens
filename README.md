# UrbanLens

GIS + Python + remote sensing + AI → evidence-based urban recommendations and interactive maps.

```bash
pip install -r requirements.txt
python urbanlens.py demo --out out_demo                     # offline, synthetic city (verified)
python urbanlens.py osm "Place, Country" --out out_place    # real OSM data (needs internet)
python urbanlens.py osm "Place" --socio zones.gpkg --ndvi ndvi.tif --lst lst.tif --llm
```
Outputs: `dashboard.html` (self-contained interactive dashboard; rebuild any time with `python dashboard.py out_dir`), `map.html`, `report.html`, `cells.geojson`, `junctions.geojson`, `recommendations.csv`, `equity.csv`, `summary.json`.

## What it measures (all scores 0-100, higher = better)
| Module | Method |
|---|---|
| Walkability | 500 m intersection density, **network pedshed ratio** (800 m), distinct everyday destination types within 10-min walk |
| Junction safety | Per-junction screening risk: pedestrian severe-injury curve vs speed × road width × legs × crossing/sidewalk mitigation; aggregated to cells |
| Pedestrian comfort | Footway coverage, low-speed share, arterial exposure, shade (vegetation) |
| Public transport | PTAL-inspired index: walk time + wait time → equivalent doorstep frequency, by mode |
| Green access | Network distance to ≥0.5 ha green space (3-30-300 rule), vegetation cover, park m² per person |
| Heat / climate | IPCC-style exposure (LST, imperviousness, vegetation) + sensitivity (deprivation, elderly, density) − adaptive capacity |
| Equity | Population-weighted quintile gaps and **concentration index** per indicator; need × deficit priority index |

## AI layer
1. **Typology discovery** – k-means on indicators, auto-labelled ("weak walkability, weak green access").
2. **Recommendation engine** – priority cells → connected priority zones → 10 intervention rules triggered by zone metrics; every recommendation lists the numbers behind it, evidence rationale, cost class, horizon, and junction IDs.
3. **Planning brief** – template by default; `--llm` (set `ANTHROPIC_API_KEY`) asks Claude to write it from the computed JSON only.

## Inputs for real use
- **Socio-economic zones** (`--socio`): polygons with `population`, `deprivation` (0-1, higher = worse), `elderly_share`. Without it equity results are flagged as meaningless.
- **Remote sensing**: any NDVI / LST GeoTIFFs; `fetch_landsat.py` builds them from Landsat (untested here). Without them, land-cover proxies are used.
- **Transit**: OSM stops get *placeholder* frequencies. Replace `trips_per_hr` from GTFS for real results.
- **Crash data**: not yet wired in – the best next step for calibrating junction risk.

## Honest limitations
Tested end-to-end only on the synthetic city (incl. manual checks of network distances). The OSM loader and Landsat script were written against library docs but not run offline. Junction risk is a heuristic screen, not a crash model. Rule-based recommendations are decision support, not designs.

## Real-city options
`python urbanlens.py osm "Place" [--bbox W,S,E,N] [--pop worldpop.tif] [--socio zones.gpkg] [--ndvi f.tif --lst f.tif] [--cell 250]`
`UrbanLens_Rishikesh.ipynb` is a ready-to-run Google Colab notebook. The OSM loader was verified offline against a generated OSM XML file (graph, speeds in mph/km/h, crossings, stops, parks, buildings) - but not yet against live Overpass servers.

## Happy Streets (street-segment scoring)
Streets are cut into 25-100 m segments scored on pedestrian infrastructure, traffic exposure, shade, lighting, vitality, stayability
(open data) plus perceived safety and accessibility (**field audit only**). Every score is tagged audit / osm / remote / proxy / default.
Outputs: Happy Streets Index map, corridor priorities, evidence-linked interventions, a before/after simulator (MODELLED, assumed effects),
a pre-filled Excel audit sheet and a KoboToolbox form. See `FIELD_AUDIT_GUIDE.md`.
