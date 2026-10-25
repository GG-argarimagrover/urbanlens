"""Rishikesh / 'Living Ganga' look for the Streamlit app: palette, banner illustration, CSS."""
from urllib.parse import quote

TEAL, FOREST, MIST, SAND, SAFFRON = "#164E50", "#527A62", "#F3F5F2", "#E8C68A", "#E87545"
BRAND = dict(
    name="UrbanLens",
    tagline="Living Ganga · Rishikesh Urban Intelligence & Happy Streets",
    sub="Walkability · Heat · Junction safety · Transit · Green space · Equity - for any city",
)


def banner_svg():
    """Himalayan foothills, the Ganga, a Laxman-Jhula-style suspension footbridge and a low sun. Pure SVG, no images."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 230" preserveAspectRatio="xMidYMid slice">
<defs>
 <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{TEAL}"/><stop offset=".55" stop-color="#2d6b66"/><stop offset="1" stop-color="{SAND}"/></linearGradient>
 <linearGradient id="river" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3f8a84"/><stop offset="1" stop-color="{TEAL}"/></linearGradient>
</defs>
<rect width="1200" height="230" fill="url(#sky)"/>
<circle cx="930" cy="128" r="40" fill="{SAFFRON}" opacity=".92"/><circle cx="930" cy="128" r="62" fill="{SAFFRON}" opacity=".16"/>
<path d="M0 150 L90 98 L170 132 L270 70 L360 126 L450 90 L560 140 L660 84 L760 130 L860 96 L980 142 L1090 88 L1200 128 L1200 230 L0 230Z" fill="#6fa79c" opacity=".55"/>
<path d="M0 170 L110 122 L190 152 L300 104 L400 150 L520 112 L640 158 L740 118 L860 160 L990 120 L1100 156 L1200 130 L1200 230 L0 230Z" fill="{FOREST}" opacity=".92"/>
<path d="M0 196 L140 160 L260 188 L400 150 L540 190 L690 158 L820 192 L960 160 L1090 190 L1200 168 L1200 230 L0 230Z" fill="#2f5d4f"/>
<rect x="0" y="188" width="1200" height="42" fill="url(#river)"/>
<g stroke="#bfe0d8" stroke-width="1.6" fill="none" opacity=".55" stroke-linecap="round">
 <path d="M30 202 q22-7 44 0 t44 0 t44 0"/><path d="M330 214 q22-7 44 0 t44 0 t44 0"/><path d="M640 203 q22-7 44 0 t44 0 t44 0"/><path d="M900 215 q22-7 44 0 t44 0 t44 0"/><path d="M1090 204 q22-7 44 0 t44 0"/>
</g>
<g stroke="#10302f" fill="none" stroke-linecap="round">
 <path d="M250 188 H760" stroke-width="3.2"/>
 <path d="M300 188 V122 M700 188 V122" stroke-width="5"/>
 <path d="M300 122 Q500 196 700 122" stroke-width="2.2"/>
 <path d="M250 188 Q275 150 300 122 M760 188 Q735 150 700 122" stroke-width="2.2"/>
 <g stroke-width="1" opacity=".8"><path d="M340 188 V158 M380 188 V168 M420 188 V176 M460 188 V181 M500 188 V183 M540 188 V181 M580 188 V176 M620 188 V168 M660 188 V158"/></g>
 <path d="M296 122 h8 M696 122 h8" stroke-width="3"/>
</g>
</svg>'''


def css():
    svg = quote(banner_svg(), safe="")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {MIST}; }}
[data-testid="stSidebar"] {{ background: #E6ECE6; border-right: 3px solid {SAND}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
.lg-banner {{ position: relative; border-radius: 16px; overflow: hidden; padding: 26px 30px 54px 30px; margin-bottom: 18px;
  background: {TEAL} url("data:image/svg+xml,{svg}") center bottom / cover no-repeat; box-shadow: 0 2px 10px rgba(22,78,80,.25); }}
.lg-banner.compact {{ padding: 16px 26px 38px 26px; }}
.lg-banner .k {{ color: {SAND}; font-size: 12px; letter-spacing: .18em; text-transform: uppercase; font-weight: 700; }}
.lg-banner h1 {{ color: #fff; margin: 4px 0 2px; padding: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.01em; line-height: 1.15; }}
.lg-banner.compact h1 {{ font-size: 1.55rem; }}
.lg-banner .s {{ color: #e8efe9; font-size: .92rem; max-width: 760px; }}
[data-testid="stMetric"] {{ background: #fff; border-radius: 12px; padding: 10px 12px; border-left: 5px solid {FOREST}; box-shadow: 0 1px 3px rgba(22,78,80,.12); }}
[data-testid="stMetricLabel"] {{ color: #4d5f5a; }}
[data-testid="stMetricValue"] {{ color: {TEAL}; font-weight: 800; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {SAND}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {TEAL}; }}
.stTabs [aria-selected="true"] {{ color: {TEAL}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {SAFFRON}; height: 3px; }}
.stButton > button[kind="primary"], .stDownloadButton > button {{ border-radius: 10px; font-weight: 700; }}
.stButton > button[kind="primary"] {{ background: {TEAL}; border-color: {TEAL}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #0f3a3c; border-color: #0f3a3c; }}
div[data-testid="stExpander"] {{ border-radius: 12px; border-color: #d3dccf; background: #fff; }}
h2, h3 {{ color: {TEAL}; }}
.lg-foot {{ margin-top: 30px; padding-top: 10px; border-top: 2px solid {SAND}; color: #5d6b66; font-size: 12px; }}
</style>"""


def banner_html(title=None, sub=None, compact=False):
    t = title or BRAND["name"]
    s = sub if sub is not None else BRAND["sub"]
    cls = "lg-banner compact" if compact else "lg-banner"
    return f'<div class="{cls}"><div class="k">{BRAND["tagline"]}</div><h1>{t}</h1><div class="s">{s}</div></div>'


def footer_html():
    return ('<div class="lg-foot">Open data © OpenStreetMap contributors · satellite layers optional · '
            'Screening tool - verify findings on the ground before investment decisions.</div>')
