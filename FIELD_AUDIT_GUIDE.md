# Happy Streets field audit - quick guide

**Why audit?** Open maps cannot tell you footpath width, obstructions, shade at noon, working lights, or whether a street *feels*
safe. Without an audit those scores are estimates (marked *proxy* / *default*) and perceived safety + accessibility are not scored at all.

## 1. Choose where to audit
In the app: **Happy Streets -> Field audit kit -> Audit sheet (Excel)**. It lists the segments where an audit changes the picture most
(busy, low-scoring, not yet audited). Start with 100; add more later. **Keep the downloaded sheet/GeoJSON** - segment IDs belong to that data snapshot.

## 2. Collect (pick one)
- **Phone (recommended):** upload `happy_streets_audit_form.xlsx` to KoboToolbox (kobotoolbox.org) -> Deploy -> collect offline with the Kobo Collect app.
  Stand at the middle of the segment, capture GPS, fill in. IDs are optional: the app matches by GPS if the ID is blank.
- **Paper/Excel:** fill the `Audit` sheet in the template; the `Codes` sheet explains every rating.

## 3. Rules that keep data comparable
- Audit each segment **twice if you can**: daytime (shade, footpath, frontage, seating) and after dark (lighting, night safety).
- Two people per segment for perceived-safety ratings (include at least one woman auditor); the tool uses the median of repeat audits.
- Rate what you observe, not what you assume. Leave a field **blank** if you could not judge it - blank is better than a guess.
- *Perceived safety here is an auditor rating, not a resident survey.* For real perception data run short intercept surveys and treat them separately.
- Record clear width at the **narrowest** point of the walkable path. Obstruction codes: 0 none, 1 minor, 2 major, 3 blocked (people forced onto the road).
- Take a photo of every segment rated 1-2 on anything (evidence for design teams).

## 4. Load results
Export from Kobo (CSV or Excel) or save your Excel sheet, then in the app upload it under **Happy Streets -> Upload completed field audits**.
The app reports how many rows matched by ID / by GPS, ignores out-of-range values with a warning, and recalculates scores. Audited segments show
**High confidence**; audit-only indicators (perceived safety, accessibility) appear for them.

## Thresholds you should set to your local standard
Footpath width, shaded-path share, seating density and active-frontage targets are assumptions in `segments.py -> PARAMS` (e.g. align width with IRC:103).
