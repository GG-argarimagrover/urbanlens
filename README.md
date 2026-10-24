# UrbanLens - understand any city

Live Streamlit app: type a city (or a bounding box), optionally upload your own data, get walkability, heat/climate
vulnerability, junction safety & comfort, public-transport connectivity, green-space access, neighbourhood equity,
AI-generated neighbourhood types and evidence-linked recommendations - with an interactive dashboard.

## Deploy (free) - same flow as any Streamlit app
1. Create a GitHub repository and upload **all files in this folder** (keep `.streamlit/` and `requirements.txt`).
2. Go to https://share.streamlit.io -> **Create app** -> pick the repo, branch `main`, main file path `app.py` -> **Deploy**.
3. First build takes a few minutes (geospatial libraries). Your app gets a URL like `https://<name>.streamlit.app`.

Optional: for the AI-written planning brief, add `ANTHROPIC_API_KEY = "..."` under app **Settings -> Secrets**.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Always from OpenStreetMap (automatic): streets, junctions, crossings, bus/tram/rail stops, shops & services, parks, buildings, trees.
Optional uploads (sidebar):
| Upload | Format | Required content |
|---|---|---|
| Ward / zone polygons | .gpkg, .geojson or zipped shapefile | columns `population`, `deprivation` (0-1, higher = worse); optional `elderly_share` (0-1) |
| Population raster | GeoTIFF | persons per pixel (e.g. WorldPop) |
| NDVI raster | GeoTIFF, 1 band | vegetation index -1..1 |
| Land-surface temperature | GeoTIFF, 1 band | e.g. Landsat 8/9 ST in deg C (`fetch_landsat.py` builds NDVI+LST) |
Files may have different resolutions/CRS - everything is resampled onto the analysis grid. No QGIS merging needed.

## Limits & honesty
- Free hosting has limited memory/time: areas are capped (~600 km2) and the grid is auto-coarsened (<= 3000 cells). Use a bounding box for big cities.
- OpenStreetMap completeness varies; the **Data quality** tab flags thin layers. Transit frequencies from OSM are placeholders (no GTFS yet).
- Equity results need deprivation data - without it the app says so and the numbers are not meaningful.
- Junction risk is a screening heuristic, not a crash model. Slope/terrain is not modelled.
- Tested: demo city, a mocked OpenStreetMap download (graph/tags/speeds/crossings/stops/parks), ward + raster uploads, bad input, and headless app runs.
  **Not yet tested against the live OpenStreetMap servers or on Streamlit Community Cloud itself** - expect to debug first deploy.
