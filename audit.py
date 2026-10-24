"""Field-audit support: Excel template, KoboToolbox XLSForm, and the reader that merges audits into segment scores.

Every audit field has the same machine name in the Excel template, the XLSForm and the reader, so data collected on
phones (Kobo/ODK) and data typed into the Excel sheet both load the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name, label, type, constraint hint, guidance
FIELDS = [
    ("auditor", "Auditor name / ID", "text", "", "Who collected this record."),
    ("audit_date", "Audit date", "date", "", "Date of the audit."),
    ("time_of_day", "Time of day", "choice:daynight", "day / night", "Night audits are needed for lighting and night-time safety."),
    ("footpath_present", "Is there a footpath (walkable edge) on this segment?", "yesno", "1 = yes, 0 = no", "If no, leave the other footpath fields blank."),
    ("continuity", "Footpath continuity", "rating", "1-5", "5 = continuous, no breaks; 3 = occasional breaks; 1 = mostly broken or missing."),
    ("clear_width_m", "Narrowest clear walking width (m)", "decimal", "0-10", "Measure the narrowest point free of poles, vendors, parked vehicles."),
    ("surface_condition", "Surface condition", "rating", "1-5", "5 = even and well kept; 3 = uneven but passable; 1 = broken/hazardous."),
    ("obstructions", "Obstructions in walking path", "scale03", "0-3", "0 = none; 1 = minor; 2 = major (pedestrians must step around); 3 = blocked (forced onto the road)."),
    ("parking_encroachment", "Parking / vending encroaching on footpath", "scale03", "0-3", "0 = none; 1 = occasional; 2 = frequent; 3 = footpath mostly occupied."),
    ("kerb_ramps", "Kerb ramps at crossings / driveways", "yesno_na", "1 / 0 / blank", "1 = present and usable; 0 = absent; blank = no kerb on this segment."),
    ("tactile_paving", "Tactile paving present", "yesno_na", "1 / 0 / blank", "1 = present; 0 = absent; blank = not applicable."),
    ("crossing_available", "Formal pedestrian crossing available on/at end of segment", "yesno_na", "1 / 0 / blank", "Marked crossing, signal or raised table."),
    ("vehicle_speed_kmh", "Typical observed vehicle speed (km/h)", "integer", "0-120", "Rough estimate or radar/stopwatch value. Leave blank if not observed."),
    ("shade_pct", "Share of walking path in shade (%)", "integer", "0-100", "Record at the time of audit; note time in 'notes' (best at 12:00-15:00)."),
    ("seating_count", "Number of public seats on the segment", "integer", "0-50", "Benches, ledges designed for sitting, parklets."),
    ("lighting_quality", "Pedestrian lighting quality (night)", "rating", "1-5", "5 = bright, even, all working; 3 = patchy; 1 = dark / not working."),
    ("active_frontage_pct", "Share of street edge with active frontage (%)", "integer", "0-100", "Doors, shop fronts, vendors, windows with activity. Blank walls and parking = inactive."),
    ("vendors_count", "Street vendors present", "integer", "0-100", "Count at the time of audit."),
    ("perceived_safety_day", "Perceived safety - daytime (auditor rating)", "rating", "1-5", "5 = very safe, 1 = unsafe. Auditor judgement - complement with resident surveys."),
    ("perceived_safety_night", "Perceived safety - night (auditor rating)", "rating", "1-5", "As above, at night."),
    ("perceived_safety_women", "Perceived safety for women and girls (auditor rating)", "rating", "1-5", "Ideally rated by women auditors / local women's groups."),
    ("accessibility_overall", "Overall accessibility for older people & wheelchair users", "rating", "1-5", "5 = fully usable independently; 1 = not usable."),
    ("notes", "Notes", "text", "", "Anything unusual; what you would fix first."),
    ("photo_ref", "Photo reference / file name", "text", "", "One photo along the street + one of the worst barrier."),
]
NUMERIC = [f[0] for f in FIELDS if f[2] not in ("text", "date", "choice:daynight")]
RANGES = {"continuity": (1, 5), "surface_condition": (1, 5), "lighting_quality": (1, 5), "perceived_safety_day": (1, 5),
          "perceived_safety_night": (1, 5), "perceived_safety_women": (1, 5), "accessibility_overall": (1, 5),
          "obstructions": (0, 3), "parking_encroachment": (0, 3), "clear_width_m": (0, 10), "vehicle_speed_kmh": (0, 120),
          "shade_pct": (0, 100), "active_frontage_pct": (0, 100), "seating_count": (0, 50), "vendors_count": (0, 100)}
YESNO = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}


# ---------------------------------------------------------------------------------- Excel template
def make_template_xlsx(seg, path, n=100):
    """Pre-filled audit workbook for the n segments where an audit would change the picture most."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    pick = seg.sort_values("audit_priority", ascending=False).head(n)
    wb = Workbook(); ws = wb.active; ws.title = "Audit"
    pre = ["segment_id", "street", "lat", "lon", "length_m", "why_audit_here"]
    head = pre + [f[0] for f in FIELDS]
    ws.append(head)
    for _, r in pick.iterrows():
        why = []
        if r.pct_audit < .5: why.append("not yet audited")
        if r.b_infra == "default": why.append("sidewalk not mapped")
        if r.demand >= .5: why.append("high pedestrian demand")
        if r.hsi < 40: why.append("low open-data score")
        ws.append([r.seg_id, r["name"] or r["corridor"], r.lat, r.lon, round(r.length_m), "; ".join(why)] + [None] * len(FIELDS))
    teal, mist = "164E50", "F3F5F2"
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor=teal)
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, h in enumerate(head, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = 16 if h in pre[2:] else (30 if h in ("street", "why_audit_here", "notes") else 14)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(pre)):
        for c in row:
            c.fill = PatternFill("solid", fgColor=mist)
    ws.freeze_panes = "C2"
    last = ws.max_row + 200   # allow extra rows for new records
    col = {h: ws.cell(1, i).column_letter for i, h in enumerate(head, 1)}
    def dv(kind, names, **kw):
        d = DataValidation(type=kind, allow_blank=True, showErrorMessage=True, **kw); ws.add_data_validation(d)
        for nm in names: d.add(f"{col[nm]}2:{col[nm]}{last}")
    dv("whole", [k for k, (a, b) in RANGES.items() if (a, b) == (1, 5)], operator="between", formula1=1, formula2=5)
    dv("whole", ["obstructions", "parking_encroachment"], operator="between", formula1=0, formula2=3)
    dv("whole", ["footpath_present", "kerb_ramps", "tactile_paving", "crossing_available"], operator="between", formula1=0, formula2=1)
    dv("decimal", ["clear_width_m"], operator="between", formula1=0, formula2=10)
    dv("whole", ["shade_pct", "active_frontage_pct"], operator="between", formula1=0, formula2=100)
    dv("whole", ["vehicle_speed_kmh"], operator="between", formula1=0, formula2=120)
    dv("whole", ["seating_count", "vendors_count"], operator="between", formula1=0, formula2=100)
    d = DataValidation(type="list", formula1='"day,night"', allow_blank=True); ws.add_data_validation(d); d.add(f"{col['time_of_day']}2:{col['time_of_day']}{last}")
    ws2 = wb.create_sheet("Codes")
    ws2.append(["field", "question", "allowed values", "how to rate"])
    for name, label, typ, hint, guide in FIELDS: ws2.append([name, label, hint, guide])
    for c in ws2[1]: c.font = Font(bold=True)
    for letter, w in zip("ABCD", (24, 52, 18, 100)): ws2.column_dimensions[letter].width = w
    ws3 = wb.create_sheet("Instructions")
    for line in ["HOW TO USE THIS SHEET",
                 "1. Each row is a ~50 m street segment (see segment_id, street, lat/lon). Walk the segment end to end.",
                 "2. Fill only what you actually observed. Leave unknown cells BLANK - never guess.",
                 "3. See the 'Codes' sheet for how to rate each field. Take 2 photos per segment and note the file name.",
                 "4. Audit in daytime AND at night where possible (lighting and perceived-safety fields).",
                 "5. Do not rename columns or change segment_id. Save as .xlsx (or export the sheet to .csv) and upload it in the app.",
                 "6. To add a street the list does not include, add a row with lat/lon filled and leave segment_id blank - the app matches it to the nearest segment (within 40 m).",
                 "7. Ratings of perceived safety are auditor judgements. For defensible results also run resident surveys, including women, older people and persons with disabilities."]:
        ws3.append([line])
    ws3.column_dimensions["A"].width = 130
    wb.move_sheet("Instructions", offset=-2)
    wb.save(path)
    return path


# ---------------------------------------------------------------------------------- XLSForm (KoboToolbox / ODK)
def make_xlsform(path):
    from openpyxl import Workbook
    wb = Workbook(); sv = wb.active; sv.title = "survey"
    sv.append(["type", "name", "label", "hint", "required", "constraint", "constraint_message", "appearance"])
    sv.append(["text", "segment_id", "Segment ID (from the printed list; leave blank for a new street)", "e.g. S00123", "no", "", "", ""])
    sv.append(["geopoint", "location", "Stand at the middle of the segment and capture location", "", "yes", "", "", ""])
    for name, label, typ, hint, guide in FIELDS:
        h = (guide + (f" [{hint}]" if hint else "")).strip()
        if typ == "text": t, extra = "text", ("", "", "")
        elif typ == "date": t, extra = "date", ("", "", "")
        elif typ == "choice:daynight": t, extra = "select_one daynight", ("", "", "")
        elif typ == "rating": t, extra = "select_one rating5", ("", "", "")
        elif typ == "scale03": t, extra = "select_one scale03", ("", "", "")
        elif typ in ("yesno", "yesno_na"): t, extra = "select_one yesno", ("", "", "")
        elif typ == "decimal": t, extra = "decimal", (f". >= 0 and . <= 10", "Enter 0-10 m", "")
        else:
            lo, hi = RANGES.get(name, (0, 100)); t, extra = "integer", (f". >= {lo} and . <= {hi}", f"Enter {lo}-{hi}", "")
        req = "yes" if name in ("footpath_present", "time_of_day") else "no"
        rel = ""
        sv.append([t, name, label, h, req, extra[0], extra[1], ""])
    sv.append(["image", "photo_street", "Photo along the street", "", "no", "", "", ""])
    sv.append(["image", "photo_barrier", "Photo of the worst barrier", "", "no", "", "", ""])
    ch = wb.create_sheet("choices"); ch.append(["list_name", "name", "label"])
    for v, lab in ((1, "1 - very poor"), (2, "2 - poor"), (3, "3 - fair"), (4, "4 - good"), (5, "5 - very good")): ch.append(["rating5", v, lab])
    for v, lab in ((0, "0 - none"), (1, "1 - minor"), (2, "2 - major"), (3, "3 - blocked / severe")): ch.append(["scale03", v, lab])
    for v, lab in ((1, "Yes"), (0, "No")): ch.append(["yesno", v, lab])
    for v, lab in (("day", "Day"), ("night", "Night")): ch.append(["daynight", v, lab])
    st = wb.create_sheet("settings"); st.append(["form_title", "form_id", "version"])
    st.append(["Happy Streets segment audit", "happy_streets_audit", "1"])
    wb.save(path)
    return path


# ---------------------------------------------------------------------------------- reading / matching
def read_audit(file):
    """Load csv/xlsx (Excel template or Kobo export). Returns (clean DataFrame, list of warnings)."""
    warn = []
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(file)
        df = xl.parse("Audit" if "Audit" in xl.sheet_names else xl.sheet_names[0])
    else:
        df = pd.read_csv(file)
    df.columns = [str(c).strip().lower().split("/")[-1] for c in df.columns]
    ren = {"_location_latitude": "lat", "_location_longitude": "lon", "latitude": "lat", "longitude": "lon"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})
    if "segment_id" in df:
        df["segment_id"] = df["segment_id"].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "": np.nan, "NONE": np.nan})
    for c in NUMERIC:
        if c in df:
            s = df[c]
            if s.dtype == object:
                s = s.astype(str).str.strip().str.lower().map(lambda v: YESNO.get(v, v))
            df[c] = pd.to_numeric(s, errors="coerce")
            if c in RANGES:
                lo, hi = RANGES[c]
                bad = df[c].notna() & ~df[c].between(lo, hi)
                if bad.any():
                    warn.append(f"{int(bad.sum())} value(s) in '{c}' outside {lo}-{hi} were ignored")
                    df.loc[bad, c] = np.nan
    has = [c for c in NUMERIC if c in df]
    df = df.dropna(subset=has, how="all") if has else df
    if not has:
        warn.append("No recognised audit columns found - use the template or XLSForm column names.")
    return df, warn


def aggregate_audit(df, seg, max_dist_m=40):
    """Match audit rows to segments (by segment_id, else nearest segment to lat/lon) and take the median per segment."""
    import geopandas as gpd
    import shapely
    df = df.copy()
    ids = set(seg.seg_id)
    if "segment_id" in df:
        df["_seg"] = df["segment_id"].astype(object).where(df["segment_id"].isin(ids), None)
    else:
        df["_seg"] = pd.Series([None] * len(df), index=df.index, dtype=object)
    by_id = int(df["_seg"].notna().sum())
    by_geo = 0
    if {"lat", "lon"} <= set(df.columns):
        need = df["_seg"].isna() & df["lat"].notna() & df["lon"].notna()
        if need.any():
            pts = gpd.GeoSeries(gpd.points_from_xy(df.loc[need, "lon"], df.loc[need, "lat"]), crs=4326).to_crs(seg.crs)
            tree = shapely.STRtree(seg.geometry.values)
            idx, dist = tree.query_nearest(pts.values, max_distance=max_dist_m, return_distance=True, all_matches=False)
            rows = df.index[need]
            df.loc[rows[idx[0]], "_seg"] = pd.Series(seg.seg_id.values[idx[1]], index=rows[idx[0]], dtype=object)
            by_geo = len(idx[0])
    matched = df.dropna(subset=["_seg"])
    cols = [c for c in NUMERIC if c in matched]
    agg = matched.groupby("_seg")[cols].median()
    agg.columns = ["a_" + c for c in agg.columns]
    rep = dict(rows=len(df), matched_by_id=by_id, matched_by_location=by_geo,
               unmatched=int(len(df) - len(matched)), segments_audited=int(len(agg)))
    return agg, rep
