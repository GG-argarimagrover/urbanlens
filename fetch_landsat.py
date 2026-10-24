#!/usr/bin/env python3
"""Build summer-median NDVI and land-surface-temperature GeoTIFFs from Landsat C2 L2 (Planetary Computer, free).
NOTE: written to the documented STAC/stackstac APIs but NOT executed in the sandbox this repo was built in
(no internet there) - treat as a starting point and check the output rasters visually.

    python fetch_landsat.py 'minLon,minLat,maxLon,maxLat' 2025-06-01 2025-08-31 out_dir
then:  python urbanlens.py osm "Place" --ndvi out_dir/ndvi.tif --lst out_dir/lst.tif
"""
import os, sys
import numpy as np, planetary_computer as pc, pystac_client, stackstac

bbox = [float(v) for v in sys.argv[1].split(",")]
t0, t1, out = sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
items = cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=f"{t0}/{t1}",
                   query={"eo:cloud_cover": {"lt": 20}, "platform": {"in": ["landsat-8", "landsat-9"]}}).item_collection()
print(len(items), "scenes")
da = stackstac.stack(items, assets=["red", "nir08", "lwir11", "qa_pixel"], bounds_latlon=bbox, resolution=30,
                     epsg=int(items[0].properties["proj:epsg"]), fill_value=np.nan, rescale=False).astype("float32")
clear = (da.sel(band="qa_pixel").astype("uint16") & 0b11000) == 0            # bits 3,4 = cloud, cloud shadow
red = da.sel(band="red") * 0.0000275 - 0.2
nir = da.sel(band="nir08") * 0.0000275 - 0.2
ndvi = ((nir - red) / (nir + red)).where(clear).median("time")
lst = (da.sel(band="lwir11") * 0.00341802 + 149.0 - 273.15).where(clear).median("time")  # deg C
for name, arr in (("ndvi", ndvi), ("lst", lst)):
    arr = arr.rio.write_crs(f"EPSG:{items[0].properties['proj:epsg']}")
    arr.compute().rio.to_raster(os.path.join(out, f"{name}.tif"))
    print("wrote", name)
