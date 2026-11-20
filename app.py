"""
AccidentIQ - Traffic Accident Severity Prediction Dashboard (Streamlit)

Run:  streamlit run app.py
Needs Streamlit >= 1.46 for the real top menu (older versions fall back to the sidebar menu).

Folder layout:
    app.py
    requirements.txt              (see the file: streamlit, pandas, plotly, scikit-learn, lightgbm, imbalanced-learn,
                                   joblib, statsmodels)
    data/                         (CSV files exported by the notebook, plus best_model.joblib, model_settings.json
                                   and Monthly_Accident_Counts.csv for the Predict and Forecast pages)

THEME: there is ONE theme control - Streamlit's own  (top-right menu) > Settings > Theme.
The dashboard follows it automatically and instantly: the page colours are written in CSS that
adapts to the active theme, and the charts use Streamlit's Plotly theme. There is no second
Light/Dark switch inside the app any more, and nothing needs a rerun when the theme changes.
Do NOT add a [theme] section to .streamlit/config.toml if you want the user to be able to pick.
"""
import html
import inspect
import json
import time
import warnings
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ================================================================= setup
st.set_page_config(page_title="AccidentIQ - Accident Severity Prediction", page_icon="🚦", layout="wide",
                   initial_sidebar_state="auto")

HERE = Path(__file__).parent
SEV_LABEL = {1: "Severity 1", 2: "Severity 2", 3: "Severity 3", 4: "Severity 4"}
SEV_ORDER = list(SEV_LABEL.values())
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]

# Mid-tone colours that stay readable on BOTH the light and the dark theme
ACCENT, GREEN, ORANGE, RED, GREY, BLUE = "#2E9BD6", "#22B07D", "#F59E0B", "#EF4444", "#8A94A6", "#4F6BED"
CYAN = ACCENT
SEV_COLOR = {"Severity 1": GREEN, "Severity 2": CYAN, "Severity 3": ORANGE, "Severity 4": RED}
NEUTRAL = "rgba(128,128,128,.18)"       # tracks / land / grid: a translucent grey works on any background
ACCENT_SCALE = [[0, "rgba(46,155,214,.10)"], [1, ACCENT]]                       # low = almost transparent
HEAT_SCALE = [[0, "rgba(245,158,11,.08)"], [0.55, ORANGE], [1, RED]]

# Short plain-language explanations shown as "?" tooltips
HELP = {
    "recall": "Out of all REAL accidents of a severity level, the share the model correctly spotted. "
              "Averaged over the four levels, so rare levels count as much as common ones.",
    "f1": "One balanced score that combines 'how often an alert is right' (precision) and 'how many real cases "
          "are caught' (recall). Averaged over the four levels. Higher is better.",
    "auc": "How well the model ranks riskier accidents above safer ones. 0.5 = pure guessing, 1.0 = perfect.",
    "acc": "Share of all predictions that were exactly right. It can look high simply because most accidents "
           "belong to one common level.",
    "within": "Share of predictions that were exactly right or only one severity level away (e.g. said 'High', was 'Moderate').",
}

# ================================================================= styling (theme-agnostic CSS)
# Every colour below is derived from the CURRENT text colour (currentColor) or is a translucent tint,
# so the same CSS looks right on Streamlit's light and dark themes without Python knowing which is active.
CSS = """
<style>
:root{
  --accent:#2E9BD6; --accent2:#22B07D; --warn:#F59E0B;
  --card:rgba(128,128,128,.08); --line:rgba(128,128,128,.28); --tint:rgba(46,155,214,.16);
  --shadow:rgba(0,0,0,.14);
}
@supports (color: color-mix(in srgb, red 50%, blue)){
  :root{
    --card:color-mix(in srgb, currentColor 6%, transparent);
    --line:color-mix(in srgb, currentColor 16%, transparent);
    --tint:color-mix(in srgb, var(--accent) 18%, transparent);
  }
}
/* App shell: keep Streamlit's own background (so the theme switch works) and add a soft accent glow */
.stApp{background-image:radial-gradient(1100px 520px at 88% -8%, rgba(46,155,214,.16), transparent 62%),
                        radial-gradient(900px 480px at -6% 4%, rgba(34,176,125,.09), transparent 60%);
       background-attachment:fixed;}
[data-testid="stDecoration"],[data-testid="stAppDeployButton"],.stAppDeployButton,footer{display:none !important;}
.block-container{padding-top:5.2rem;padding-bottom:3rem;max-width:1440px;}
html{scroll-behavior:smooth;}
h1,h2,h3,h4{letter-spacing:-.02em;}
.muted{opacity:.72;}

/* Brand bar */
.topbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin:0 0 18px;
        padding:4px 0 14px;border-bottom:1px solid var(--line);}
.brand{font-size:1.3rem;font-weight:800;display:flex;align-items:center;gap:10px;letter-spacing:-.02em;}
.brand .logo{display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:12px;
             background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 6px 16px rgba(46,155,214,.35);}
.brand b{color:var(--accent);}
.tagline{font-size:.85rem;opacity:.7;}

/* Hero */
.hero{position:relative;overflow:hidden;padding:30px 34px;border-radius:22px;margin-bottom:22px;
      background:linear-gradient(125deg,rgba(46,155,214,.22),rgba(34,176,125,.10) 60%,rgba(46,155,214,.06));
      border:1px solid rgba(46,155,214,.38);box-shadow:0 12px 34px var(--shadow);}
.hero h1{font-size:clamp(1.7rem,3vw,2.5rem);margin:10px 0 8px;line-height:1.14;font-weight:800;padding:0;}
.hero p{font-size:1.02rem;margin:0;max-width:900px;line-height:1.6;opacity:.85;}
.tag{display:inline-flex;align-items:center;font-size:.7rem;letter-spacing:1.4px;font-weight:700;color:var(--accent);
     border:1px solid rgba(46,155,214,.55);border-radius:999px;padding:5px 11px;background:rgba(46,155,214,.08);}
.tag::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--accent);margin-right:8px;}

/* Cards and metric blocks */
[data-testid="stMetric"],[data-testid="stPlotlyChart"],[data-testid="stAlert"],.glass,.step,[data-testid="stExpander"]{
  background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 18px var(--shadow);
  transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;}
[data-testid="stMetric"]{padding:16px 18px;}
[data-testid="stMetricValue"]{color:var(--accent);font-weight:750;}
[data-testid="stPlotlyChart"]{padding:8px;}
[data-testid="stMetric"]:hover,.glass:hover,.step:hover{transform:translateY(-2px);border-color:rgba(46,155,214,.55);}
.glass{padding:18px 20px;height:100%;margin-bottom:1rem;}   /* the 1rem cancels Streamlit's negative bottom margin on markdown blocks */
.glass h4{margin:0 0 8px;color:var(--accent);font-size:1rem;padding:0;}
.glass p{margin:0;font-size:.92rem;line-height:1.55;opacity:.9;}
.insight .lbl{font-size:.78rem;letter-spacing:.8px;text-transform:uppercase;opacity:.65;font-weight:650;}
.insight .big{font-size:2.1rem;font-weight:800;color:var(--accent);line-height:1.15;margin:4px 0 6px;}
.callout{padding:16px 20px;border-radius:16px;margin:8px 0 18px;font-size:1rem;line-height:1.55;
         background:linear-gradient(90deg,rgba(34,176,125,.20),rgba(34,176,125,.05));border:1px solid rgba(34,176,125,.5);}
.callout.warn{background:linear-gradient(90deg,rgba(245,158,11,.20),rgba(245,158,11,.05));border-color:rgba(245,158,11,.55);}

/* Tables */
.tbl-wrap{overflow-x:auto;border-radius:14px;background:var(--card);border:1px solid var(--line);
          box-shadow:0 4px 18px var(--shadow);margin-bottom:8px;}
.tbl{width:100%;border-collapse:collapse;font-size:.92rem;}
.tbl th,.tbl td{padding:11px 14px;text-align:right;white-space:nowrap;}
.tbl th:first-child,.tbl td:first-child{text-align:left;}
.tbl thead th{background:var(--tint);font-weight:700;}
.tbl tbody th{font-weight:600;}
.tbl tbody tr + tr th,.tbl tbody tr + tr td{border-top:1px solid var(--line);}
.tbl td.best{background:var(--tint);font-weight:700;color:var(--accent);}

/* Workflow */
.steps{display:flex;gap:10px;align-items:stretch;margin:8px 0 20px;}
.step{flex:1;padding:14px;text-align:center;}
.step b{color:var(--accent);display:block;margin-bottom:5px;}
.step span{font-size:.84rem;line-height:1.4;opacity:.85;}
.arrow{align-self:center;color:var(--accent);font-size:1.1rem;}
hr{border-color:var(--line);}
button:active{transform:scale(.98);}
[data-baseweb="select"] > div:hover{border-color:var(--accent) !important;}

/* Loading */
.loader{max-width:480px;margin:32px auto;padding:24px;border-radius:16px;text-align:center;background:var(--card);border:1px solid var(--line);}
.road{position:relative;height:36px;overflow:hidden;border-bottom:2px dashed var(--line);}
.car{position:absolute;bottom:2px;font-size:1.6rem;animation:drive 1.7s linear infinite;}
@keyframes drive{0%{left:-12%;}100%{left:104%;}}
.loader-title{margin-top:12px;font-weight:700;}
.loader-sub{opacity:.65;font-size:.84rem;}
[data-testid="stProgress"] [role="progressbar"] > div{background:linear-gradient(90deg,#2E9BD6,#22B07D) !important;}

/* Animated score bars, legend chips, pop-in numbers */
.bars{display:flex;flex-direction:column;gap:16px;margin:10px 0 8px;}
.bar-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;font-size:.96rem;}
.bar-row .name{font-weight:700;}
.bar-row .desc{opacity:.65;font-size:.82rem;margin-left:8px;}
.bar-row .val{font-weight:800;font-variant-numeric:tabular-nums;}
.bar-row.pick .name{color:var(--accent);}
.track{position:relative;height:16px;border-radius:999px;background:var(--card);border:1px solid var(--line);margin-top:6px;}
.fill{height:100%;width:0;border-radius:999px;animation:grow 1s cubic-bezier(.2,.75,.25,1) forwards;animation-delay:var(--d,0s);}
@keyframes grow{from{width:0}to{width:var(--w)}}
.mark{position:absolute;top:-5px;bottom:-5px;width:3px;margin-left:-1px;border-radius:2px;background:currentColor;opacity:.55;}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 4px;}
.chip{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;border:1px solid var(--line);
      background:var(--card);font-size:.86rem;}
.chip i{width:10px;height:10px;border-radius:50%;display:inline-block;}
.insight .big{animation:pop .65s cubic-bezier(.2,.9,.3,1.25) backwards;}
@keyframes pop{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:none}}
.loader-title{animation:pulse 1.6s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* Equal-height cards: every card in a row stretches to the tallest one */
[data-testid="stHorizontalBlock"]{align-items:stretch !important;}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]{height:100%;}
[data-testid="stElementContainer"]:has(.glass),
[data-testid="stElementContainer"]:has(> [data-testid="stMetric"]){flex:1 1 auto;}
[data-testid="stElementContainer"]:has(.glass){display:flex;flex-direction:column;}
[data-testid="stElementContainer"]:has(.glass) > *,
[data-testid="stElementContainer"]:has(.glass) > * > *,
[data-testid="stElementContainer"]:has(.glass) > * > * > *{flex:1 1 auto;display:flex;flex-direction:column;}
[data-testid="stMetric"]{height:100%;}

/* Website-style top navigation bar: solid bar, logo left, links beside it, thin line underneath */
[data-testid="stHeader"]{
  border-bottom:1px solid var(--line);
  box-shadow:0 2px 14px var(--shadow);
}
[data-testid="stToolbar"]{padding:0 .5rem;}
[data-testid="stHeaderLogo"]{height:2rem;width:auto;}
[data-testid="stTopNavLinkContainer"]{gap:2px;}

/* Top navigation links: quiet text links, accent underline on the current page */
[data-testid="stTopNavLink"]{
  background:transparent !important;
  border-radius:8px !important;
  font-size:.92rem !important;
  font-weight:600 !important;
  padding:.45rem .85rem !important;
  opacity:.72;
  transition:opacity .15s ease, background .15s ease !important;
}
[data-testid="stTopNavLink"]:hover{
  opacity:1;
  background:var(--tint) !important;
}
[data-testid="stTopNavLink"][aria-current="page"]{
  opacity:1;
  background:transparent !important;
  color:var(--accent) !important;
  box-shadow:inset 0 -2px 0 var(--accent) !important;
  border-radius:0 !important;
}

[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: var(--tint) !important;
}

/* Page-in animation (each page has its own keyframe name so it replays when you switch pages) */
__ANIM__

@media(max-width:760px){
 .block-container{padding-left:1rem;padding-right:1rem;padding-top:4.6rem;}
 .hero{padding:22px 20px;border-radius:18px;}
 .hero h1{font-size:1.65rem;}
 .hero p{font-size:.94rem;}
 .steps{flex-direction:column;}
 .arrow{transform:rotate(90deg);}
}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important;}
 .fill{width:var(--w) !important;}}
</style>
"""


def page_anim_css():
    out = []
    for key, kf in (("overview", "riseA"), ("explore", "riseB"), ("training", "riseC"), ("test", "riseD"),
                    ("predict", "riseE"), ("forecast", "riseF")):
        out.append(f"@keyframes {kf}{{from{{opacity:0;transform:translateY(22px)}}to{{opacity:1;transform:none}}}}")
        out.append(f".st-key-page_{key}>*{{animation:{kf} .6s cubic-bezier(.2,.75,.25,1) backwards}}")
        for i in range(2, 21):
            out.append(f".st-key-page_{key}>*:nth-child({i}){{animation-delay:{min((i - 1) * 60, 800)}ms}}")
    return "".join(out)


st.markdown(CSS.replace("__ANIM__", page_anim_css()), unsafe_allow_html=True)

LOADER_SECONDS = 1.6      # how long the first-load animation plays at least (set to 0 to skip the pause)
SEV_PLAIN = {1: "short delay", 2: "moderate delay", 3: "long delay", 4: "very long delay"}


def loader_html(title, sub=""):
    return ('<div class="loader"><div class="road"><span class="car">🚗</span></div>'
            f'<div class="loader-title">{html.escape(title)}</div><div class="loader-sub">{html.escape(sub)}</div></div>')


LOADER = loader_html("Loading accident data…", "Reading the result files exported from the notebook")


@contextmanager
def loading(title, sub=""):
    """Show the driving-car animation while a slow step (loading the model, fitting a forecast) runs."""
    box = st.empty()
    box.markdown(loader_html(title, sub), unsafe_allow_html=True)
    try:
        yield
    finally:
        box.empty()


def sev_legend():
    chips = "".join(f'<span class="chip"><i style="background:{SEV_COLOR[SEV_LABEL[k]]}"></i>'
                    f'<b>{SEV_LABEL[k]}</b> · {SEV_PLAIN[k]}</span>' for k in SEV_LABEL)
    st.markdown(f'<div class="legend">{chips}</div>', unsafe_allow_html=True)
    st.caption("In this dataset, severity describes how much an accident disrupted traffic "
               "(1 = least, 4 = most). It is not an injury or fatality level.")


# ================================================================= helpers
def pct(x):
    return f"{x:.0%}"


def style(fig, height=380):
    """Transparent background + sizes only. Text / grid colours come from Streamlit's Plotly theme,
    so the charts switch between light and dark together with the rest of the app."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), title=dict(font=dict(size=16), x=0.02),
        margin=dict(l=14, r=14, t=56, b=14), legend=dict(orientation="h", y=-0.22, title=None),
    )
    fig.update_xaxes(gridcolor=NEUTRAL, zeroline=False)
    fig.update_yaxes(gridcolor=NEUTRAL, zeroline=False)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor=NEUTRAL, showlakes=False, subunitcolor="rgba(128,128,128,.45)")
    return fig


_NEW_PLOTLY_API = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, box=st):
    # theme="streamlit" (explicit) = charts follow the app theme automatically
    if _NEW_PLOTLY_API:
        box.plotly_chart(fig, width="stretch", theme="streamlit")
    else:  # older Streamlit
        box.plotly_chart(fig, use_container_width=True, theme="streamlit")


def html_table(df, fmt=None, highlight_max=False):
    """Small table as plain HTML, so it follows the light / dark colours."""
    fmt = fmt or {}
    maxes = df.select_dtypes("number").max() if highlight_max else {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<th scope='row'>{html.escape(str(idx))}</th>"]
        for c in df.columns:
            v = r[c]
            txt = "-" if pd.isna(v) else fmt.get(c, fmt.get("*", "{}")).format(v)
            best = " class='best'" if highlight_max and c in maxes and v == maxes[c] else ""
            cells.append(f"<td{best}>{html.escape(txt)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def hero(tag, title, sub):
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span><h1>{title}</h1><p>{sub}</p></div>',
                unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f"### {title}")
    if sub:
        st.caption(sub)


def glass(box, icon, title, text):
    box.markdown(f'<div class="glass"><h4>{icon} {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def insight(box, label, big, text):
    box.markdown(f'<div class="glass insight"><div class="lbl">{label}</div><div class="big">{big}</div>'
                 f'<p>{text}</p></div>', unsafe_allow_html=True)


def callout(text, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def keyed(key):
    """st.container with a key (gives the CSS hook for the page animation); falls back on old Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='190' height='40' viewBox='0 0 190 40'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='#2E9BD6'/><stop offset='1' stop-color='#22B07D'/></linearGradient></defs>"
    "<rect x='2' y='2' width='36' height='36' rx='10' fill='url(#g)'/>"
    "<rect x='13' y='8' width='14' height='24' rx='5' fill='#0F172A' fill-opacity='.85'/>"
    "<circle cx='20' cy='14' r='3' fill='#EF4444'/><circle cx='20' cy='20' r='3' fill='#F59E0B'/>"
    "<circle cx='20' cy='26' r='3' fill='#22B07D'/>"
    "<text x='48' y='27' font-family='Inter,Segoe UI,Arial,sans-serif' font-size='20' font-weight='800' "
    "fill='#7B8798'>Accident</text>"
    "<text x='127' y='27' font-family='Inter,Segoe UI,Arial,sans-serif' font-size='20' font-weight='800' "
    "fill='#2E9BD6'>IQ</text></svg>"
)


def brand_bar():
    """Put the brand (logo + name) into the top navigation bar itself instead of a second bar in the page."""
    from urllib.parse import quote
    uri = "data:image/svg+xml;utf8," + quote(LOGO_SVG)
    try:
        st.logo(uri, icon_image=uri, size="large", link=None)
    except Exception:  # very old Streamlit: no st.logo -> skip, the page still works
        pass


def glossary():
    with st.expander("📖 What do these words mean?"):
        st.markdown(
            "- **Severity (1 to 4)**: how much the accident disrupted traffic: 1 = short delay, 4 = very long delay. It is not an injury/fatality label.\n"
            "- **Accuracy**: share of predictions that were exactly right. It can look high just because most accidents "
            "are 'Moderate'.\n"
            "- **Recall (catch rate)**: of the real accidents at a level, how many the model spotted. "
            "Especially useful for less common severity classes.\n"
            "- **Precision**: when the model says a level, how often it is right.\n"
            "- **F1**: one score that balances precision and recall.\n"
            "- **AUC**: how well the model ranks risky cases above safe ones (0.5 = guessing, 1.0 = perfect).\n"
            "- **Average over four levels**: every severity level counts equally, even the rare ones.\n"
            "- **Cross-validation**: training data is split into 10 parts; the model learns on 9 and is checked on the "
            "10th, 10 times. This compares models fairly without touching the test data.")


def data_settings():
    with st.expander("⚙️ Data settings"):
        st.text_input("Folder with the CSV files exported from the notebook", key="folder")


# ================================================================= data
st.session_state.setdefault("folder", str(HERE / "data"))
folder = st.session_state["folder"]

if not Path(folder).exists():
    st.error(f"Folder '{folder}' not found. Copy the notebook's CSV exports into it, or change the folder below.")
    data_settings()
    st.stop()


@st.cache_data
def read(name, folder, clean=True):
    df = pd.read_csv(Path(folder) / name, low_memory=False)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")] if clean else df


FILES = [("acc", "PowerBI_Cleaned_Dataset.csv"), ("pred", "PowerBI_Test_Predictions.csv"),
         ("perf", "Tuned_Model_Performance.csv"), ("cv", "Tuning_CV_Summary.csv"),
         ("imp", "PowerBI_Feature_Importance.csv"), ("cm", "PowerBI_Confusion_Matrix.csv"),
         ("pc", "Per_Class_Performance.csv"), ("bal", "Balancing_Class_Distribution.csv")]

first_load = "data_loaded" not in st.session_state
if first_load:  # animated loader (car on a road + progress bar) on the first load of the session
    loader = st.empty()
    with loader.container():
        st.markdown(LOADER, unsafe_allow_html=True)
        bar = st.progress(0.0, text="Starting…")

data = {}
try:
    for i, (key, fname) in enumerate(FILES):
        if first_load:
            bar.progress(i / len(FILES), text=f"Reading {fname} ({i + 1} of {len(FILES)})")
        t_file = time.time()
        data[key] = read(fname, folder, clean=(key != "bal"))
        if first_load:                                    # small pause so the animation is visible
            time.sleep(max(0.0, LOADER_SECONDS / len(FILES) - (time.time() - t_file)))
except FileNotFoundError as e:
    if first_load:
        loader.empty()
    st.error(f"Missing file: {e.filename}")
    data_settings()
    st.stop()

if first_load:
    bar.progress(1.0, text="Ready")
    loader.empty()
    st.session_state["data_loaded"] = True

acc, pred, perf, cv = data["acc"], data["pred"], data["perf"], data["cv"].drop(columns=["Best Parameters"], errors="ignore")
imp, cm, pc, bal_raw = data["imp"], data["cm"], data["pc"], data["bal"]

acc["Severity Label"] = acc["Severity"].map(SEV_LABEL)
acc["Day"] = acc["Day_of_Week"].map(dict(enumerate(DAYS)))
acc["Weather_Condition"] = acc["Weather_Condition"].fillna("Unknown")

pred["Start_Time"] = pd.to_datetime(pred["Start_Time"], errors="coerce")
pred["Hour"] = pred["Start_Time"].dt.hour
pred["Weather_Condition"] = pred["Weather_Condition"].fillna("Unknown")

bal = bal_raw.rename(columns={bal_raw.columns[0]: "Severity"})
bal = bal[bal["Severity"].astype(str) != "Total"]
bal["Severity"] = bal["Severity"].astype(int)
bal = bal.melt(id_vars="Severity", var_name="Stage", value_name="Records")
bal["Severity Label"] = bal["Severity"].map(SEV_LABEL)

best_name = cv.loc[cv["Selection score"].idxmax(), "Model"]                 # chosen on training data only
best_model = pred["Best_Model"].iloc[0] if "Best_Model" in pred else best_name  # model shown on the test pages


# ================================================================= pages
def overview():
    with keyed("page_overview"):
        hero("AI · BIG DATA · ROAD SAFETY",
             "Turning accident data into severity insight",
             "AccidentIQ learns from real US accident records to predict how severe an accident is likely to be, "
             "and shows where, when and in what conditions the risk concentrates.")
        sev_legend()

        section("Where to go")
        guide = [("🔮", "Predict Severity", "Describe a new accident and see what the model expects."),
                 ("📈", "Accident Forecast", "How many accidents might be reported in a future year?"),
                 ("🔎", "Risk Explorer", "Filter by state, weather and time to see how risk changes."),
                 ("🎯", "Test Results", "Technical: how well the model did on unseen accidents."),
                 ("🧠", "Model Training", "Technical: how the three models were built.")]
        for col, (icon, title, text) in zip(st.columns(5), guide):
            glass(col, icon, title, text)

        pb = perf[perf["Model"] == best_model].iloc[0]
        sev34 = (acc["Severity"] >= 3).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accidents analysed", f"{len(acc):,}", help="Random sample of the US Accidents dataset.")
        c2.metric("States covered", acc["State"].nunique())
        c3.metric("Severity 3–4 share", pct(sev34), help="Share of records in Severity classes 3 and 4.")
        c4.metric("Best model", best_model, help="Chosen from three tuned models.")

        # ---- 1. what the data says (all numbers computed from the data) ----
        section("What the data says")
        peak_hour = int(acc["Hour"].value_counts().idxmax())
        rush_share = acc["Hour"].isin(RUSH_HOURS).mean()
        rush_txt = (f"Rush hours (7–9 am and 4–7 pm) are {7 / 24:.0%} of the day but see {pct(rush_share)} of accidents."
                    if rush_share > 7 / 24 else "The busiest hour of the day for accidents in this sample.")

        wx = (acc[acc["Weather_Condition"] != "Unknown"].groupby("Weather_Condition")
              .agg(n=("Severity", "size"), risk=("Severity", lambda s: (s >= 3).mean()))
              .query("n >= 100").sort_values("risk", ascending=False))
        state = acc["State"].value_counts().reset_index()
        state.columns = ["State", "Accidents"]

        a, b, c = st.columns(3)
        insight(a, "Busiest hour", f"{peak_hour:02d}:00", html.escape(rush_txt))
        if len(wx):
            top_wx = wx.iloc[0]
            insight(b, "Highest-risk weather", html.escape(str(wx.index[0])),
                    f"{pct(top_wx['risk'])} of its accidents are Severity 3–4, versus {pct(sev34)} overall.")
        else:
            insight(b, "Highest-risk weather", "n/a", "Not enough records per weather type.")
        insight(c, "Most accidents", html.escape(str(state.iloc[0]["State"])),
                f"{pct(state.iloc[0]['Accidents'] / len(acc))} of all accidents in the sample.")

        # ---- 2. proof ----
        section("Does it work?", "Measured on accidents the model never saw during training.")
        share = pred["Actual_Severity"].value_counts(normalize=True).max()
        base_f1 = (2 * share / (1 + share)) / 4          # macro-F1 of a model that always says the most common level
        within = ((pred["Best_Model_Predicted"] - pred["Actual_Severity"]).abs() <= 1).mean()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Balanced score (F1)", pct(pb["F1 (macro)"]), f"{(pb['F1 (macro)'] - base_f1) * 100:+.0f} pts vs always guessing",
                  help=HELP["f1"])
        p2.metric("Average catch rate (recall)", pct(pb["Recall (macro)"]), f"{(pb['Recall (macro)'] - 0.25) * 100:+.0f} pts vs guessing",
                  help=HELP["recall"])
        p3.metric("Ranking quality (AUC)", f"{pb['ROC-AUC (macro, OvR)']:.2f}", f"{pb['ROC-AUC (macro, OvR)'] - 0.5:+.2f} vs guessing (0.50)",
                  help=HELP["auc"])
        p4.metric("Right or 1 level off", pct(within), help=HELP["within"])
        callout("<b>Honest reading:</b> the model beats guessing on every measure and does best on the common severity "
                "levels. The rarest level (Severity 4) is still hard to catch, and it is shown openly on the "
                "<i>Test Results</i> page. Severity here is the dataset’s recorded impact class, not an injury or fatality label.",
                warn=True)

        # ---- 3. patterns ----
        section("Accident patterns across location, time and conditions")
        left, right = st.columns(2)
        sev = acc["Severity Label"].value_counts().reindex(SEV_ORDER).reset_index()
        sev.columns = ["Severity", "Accidents"]
        fig = px.pie(sev, names="Severity", values="Accidents", hole=0.58, color="Severity",
                     color_discrete_map=SEV_COLOR, category_orders={"Severity": SEV_ORDER},
                     title="Accidents by recorded severity level (share of all accidents)")
        fig.update_traces(textinfo="percent", textfont_size=14)
        show(style(fig), left)

        show(style(px.choropleth(state, locations="State", locationmode="USA-states", color="Accidents",
                                 scope="usa", color_continuous_scale=ACCENT_SCALE, title="Accident hotspots by state")),
             right)

        heat = acc.groupby(["Day", "Hour"]).size().unstack(fill_value=0).reindex(DAYS)
        show(style(px.imshow(heat, aspect="auto", color_continuous_scale=HEAT_SCALE,
                             labels=dict(x="Hour of day (0 = midnight)", y="", color="Accidents"),
                             title="When accidents happen (darker / redder = more accidents)"), 340))

        left, right = st.columns(2)
        wx10 = wx.head(10).iloc[::-1].reset_index()
        show(style(px.bar(wx10, x="risk", y="Weather_Condition", orientation="h", color_discrete_sequence=[ORANGE],
                          title="Weather conditions with the highest Severity 3–4 share")
                   .update_xaxes(tickformat=".0%", title="Share of Severity 3–4 accidents")), left)
        left.caption("Only weather types with at least 100 accidents.")

        top = state.head(10).iloc[::-1]
        show(style(px.bar(top, x="Accidents", y="State", orientation="h", color_discrete_sequence=[CYAN],
                          title="Top 10 states by number of accidents")), right)

        yr = acc.groupby(["Year", "Severity Label"]).size().reset_index(name="Accidents")
        show(style(px.line(yr, x="Year", y="Accidents", color="Severity Label", markers=True,
                           color_discrete_map=SEV_COLOR, category_orders={"Severity Label": SEV_ORDER},
                           title="Accidents per year, by severity"), 340))

        # ---- 4. how it works ----
        section("How it works")
        st.markdown(
            '<div class="steps">'
            '<div class="step"><b>1 Collect</b><span>Real accident records from the US Accidents dataset</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>2 Prepare</b><span>Clean the data and balance rare severity levels</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>3 Predict</b><span>Three AI models compared; the best one is chosen</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>4 Explain</b><span>This dashboard turns results into clear insight</span></div>'
            '</div>', unsafe_allow_html=True)

        # ---- 5. value + next steps ----
        section("Next steps")
        a, b, c = st.columns(3)
        glass(a, "🌦️", "Live conditions", "Feed in live weather and traffic data so the model can score conditions as they happen.")
        glass(b, "🛣️", "Richer road data", "Add road type, speed limits and traffic volume to improve the rare, serious classes.")
        glass(c, "🏙️", "Local dashboards", "Tailor views for a city, county or road agency with their own data.")


def explorer():
    with keyed("page_explore"):
        hero("RISK EXPLORER", "Explore accident risk by scenario",
             "Pick a state, weather and time of day to see how the mix of severity changes. "
             "This shows historical patterns in the sample, not a forecast.")

        c1, c2, c3, c4 = st.columns(4)
        states = c1.multiselect("State", sorted(acc["State"].dropna().unique()), placeholder="All states")
        weather_opts = acc.loc[acc["Weather_Condition"] != "Unknown", "Weather_Condition"].value_counts().head(15).index.tolist()
        weather = c2.multiselect("Weather", weather_opts, placeholder="All weather")
        hours = c3.slider("Hour of day", 0, 23, (0, 23), help="0 = midnight, 17 = 5 pm")
        days = c4.selectbox("Days", ["All days", "Weekdays", "Weekends"])

        f = acc
        if states:
            f = f[f["State"].isin(states)]
        if weather:
            f = f[f["Weather_Condition"].isin(weather)]
        f = f[f["Hour"].between(hours[0], hours[1])]
        if days == "Weekdays":
            f = f[f["Day_of_Week"] < 5]
        elif days == "Weekends":
            f = f[f["Day_of_Week"] >= 5]

        if f.empty:
            st.warning("No accidents match this scenario. Try widening the filters.")
            return
        if len(f) < 100:
            callout(f"Only <b>{len(f):,}</b> matching accidents, so treat the percentages below as rough.", warn=True)

        overall = (acc["Severity"] >= 3).mean()
        scen = (f["Severity"] >= 3).mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Matching accidents", f"{len(f):,}", f"{len(f) / len(acc):.1%} of the sample", delta_color="off")
        m2.metric("Severity 3–4 share", pct(scen), f"{(scen - overall) * 100:+.1f} pts vs all accidents",
                  delta_color="inverse", help="Higher than average is shown in red.")
        m3.metric("Most common level", SEV_LABEL[int(f["Severity"].mode().iloc[0])])

        left, right = st.columns(2)
        mix = pd.DataFrame({
            "Scenario": f["Severity Label"].value_counts(normalize=True).reindex(SEV_ORDER).fillna(0),
            "All accidents": acc["Severity Label"].value_counts(normalize=True).reindex(SEV_ORDER).fillna(0),
        }).reset_index().rename(columns={"index": "Severity", "Severity Label": "Severity"}).melt(
            id_vars="Severity", var_name="Group", value_name="Share")
        show(style(px.bar(mix, x="Severity", y="Share", color="Group", barmode="group",
                          color_discrete_sequence=[ORANGE, GREY], category_orders={"Severity": SEV_ORDER},
                          title="Severity mix: your scenario vs all accidents")
                   .update_yaxes(tickformat=".0%", title="Share of accidents")), left)

        hr = f.groupby("Hour").size().reindex(range(24), fill_value=0).reset_index(name="Accidents")
        show(style(px.area(hr, x="Hour", y="Accidents", color_discrete_sequence=[CYAN],
                           title="Accidents by hour in this scenario")
                   .update_xaxes(title="Hour of day (0 = midnight)")), right)


def training():
    with keyed("page_training"):
        hero("MODEL TRAINING", "How the models were built",
             "Everything on this page comes from the training data only (cross-validation), "
             "so the test data stays completely unseen.")
        callout(f"<b>In plain words:</b> three AI models studied thousands of past accidents. Each was tried with many settings "
                f"and checked fairly, and the best one, <b>{best_name}</b>, was kept. This is the technical page; "
                "<i>Predict Severity</i> is the simple one.")
        glossary()

        row = cv[cv["Model"] == best_name].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best_name)
        c2.metric("Balanced score (F1)", pct(row["CV F1 (macro)"]), help=HELP["f1"])
        c3.metric("Average catch rate (recall)", pct(row["CV Recall (macro)"]), help=HELP["recall"])
        c4.metric("Training examples (after balancing)",
                  f"{int(bal[bal['Stage'] == 'After SMOTENC']['Records'].sum()):,}",
                  help="Training set size after shrinking the most common level and creating extra examples for the rare ones.")

        st.markdown(
            '<div class="steps">'
            '<div class="step"><b>1 Clean</b><span>Remove duplicates and details only known after the accident</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>2 Select</b><span>Keep the most useful information</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>3 Balance</b><span>Even out common and rare severity levels</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>4 Tune</b><span>Try many settings, check with cross-validation</span></div><div class="arrow">➜</div>'
            '<div class="step"><b>5 Choose</b><span>Pick the best model by recall and F1</span></div>'
            '</div>', unsafe_allow_html=True)

        callout("<b>Why balance the data?</b> Serious accidents are rare, so a model can ignore them and still look "
                "accurate. The training data was made more even (shrink the biggest group, then create extra examples "
                "for rare levels with SMOTENC) so every severity level is learned fairly. "
                "The test data was left untouched.")

        left, right = st.columns(2)
        stage_order = ["Original training set", "After Random Undersampling", "After SMOTENC"]
        show(style(px.bar(bal, x="Severity Label", y="Records", color="Stage", barmode="group",
                          category_orders={"Severity Label": SEV_ORDER, "Stage": stage_order},
                          color_discrete_sequence=[GREY, ORANGE, CYAN],
                          title="Training data before and after balancing")), left)

        cvl = cv.rename(columns={"CV Accuracy": "Accuracy", "CV Recall (macro)": "Recall", "CV F1 (macro)": "F1"})
        cvl = cvl.melt(id_vars="Model", value_vars=["Accuracy", "Recall", "F1"], var_name="Score", value_name="Value")
        show(style(px.bar(cvl, x="Model", y="Value", color="Score", barmode="group",
                          color_discrete_sequence=[CYAN, ORANGE, GREEN], title="Cross-validation scores per model")
                   .update_yaxes(range=[0, 1], tickformat=".0%", title="Score")), right)

        st.markdown("### Model scoreboard")
        board = cv[["Model", "CV Accuracy", "CV Precision (macro)", "CV Recall (macro)", "CV F1 (macro)",
                    "Selection score"]].set_index("Model")
        board.columns = ["Accuracy", "Precision", "Recall", "F1", "Overall score"]
        html_table(board, {"*": "{:.1%}"}, highlight_max=True)
        st.caption(f"Overall score = average of recall and F1. The model with the highest overall score "
                   f"(highlighted) is chosen: **{best_name}**. Precision, recall and F1 are averaged over the four severity levels.")

        attr = imp[imp["Level"] == "Attribute"].copy()
        attr["Share"] = attr["Importance"] / attr["Importance"].sum()
        top = attr.sort_values("Share", ascending=False).head(10).iloc[::-1]
        show(style(px.bar(top, x="Share", y="Feature", orientation="h", color="Share",
                          color_continuous_scale=ACCENT_SCALE, title="Features the model relies on (top 10)")
                   .update_xaxes(tickformat=".0%", title="Share of the model's decisions")
                   .update_coloraxes(showscale=False), 400))
        first = top.iloc[-1]
        callout(f"💡 <b>{html.escape(str(first['Feature']))}</b> is the strongest signal, about {pct(first['Share'])} of the model's "
                "decisions. This shows what the model relies on, not what causes a severe accident.")

def results():
    with keyed("page_test"):
        hero("UNSEEN TEST DATA", "How well does it work on new accidents?",
             "These accidents were never used to train the models. This is the honest measure of performance.")
        callout("<b>In plain words:</b> 20% of the accidents were hidden from the models. We then asked the best model to guess "
                "their severity and compared its guesses with what really happened. Below: how often it was right, "
                "and where it struggled.")
        glossary()

        f1, f2 = st.columns(2)
        states = f1.multiselect("Show only these states", sorted(pred["State"].dropna().unique()))
        years = f2.multiselect("Show only these years", sorted(pred["Year"].dropna().unique()))
        p = pred.copy()
        if states:
            p = p[p["State"].isin(states)]
        if years:
            p = p[p["Year"].isin(years)]
        if p.empty:
            st.warning("No records match these filters.")
            return

        best = best_model
        pb = perf[perf["Model"] == best].iloc[0]

        g, k = st.columns([1, 2])
        acc_val = p["Best_Model_Correct"].mean() * 100
        base = p["Actual_Severity"].value_counts(normalize=True).max() * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acc_val, number=dict(suffix="%", font=dict(size=44)),
            title=dict(text="Overall accuracy", font=dict(size=16)),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=CYAN), bgcolor=NEUTRAL,
                       threshold=dict(line=dict(color=ORANGE, width=4), thickness=0.8, value=base))))
        show(style(gauge, 300), g)
        g.caption(f"Orange line ({base:.0f}%) = the score of a model that always guesses the most common level. "
                  "Accuracy alone can look good because most accidents share one level, so also check the catch rate below.")

        with k:
            r1 = st.columns(3)
            r1[0].metric("Test records (filtered)", f"{len(p):,}")
            r1[1].metric("Average catch rate (recall)", pct(pb["Recall (macro)"]), help=HELP["recall"])
            r1[2].metric("Balanced score (F1)", pct(pb["F1 (macro)"]), help=HELP["f1"])
            r2 = st.columns(3)
            r2[0].metric("Ranking quality (AUC)", f"{pb['ROC-AUC (macro, OvR)']:.2f}", help=HELP["auc"])
            within = ((p["Best_Model_Predicted"] - p["Actual_Severity"]).abs() <= 1).mean()
            r2[1].metric("Right or 1 level off", pct(within), help=HELP["within"])
            r2[2].metric("Best model", best)
            st.caption("Catch rate, F1 and ranking quality cover the full test set; the other figures follow the filters.")

        left, right = st.columns(2)
        pl = perf.rename(columns={"Recall (macro)": "Catch rate (recall)", "F1 (macro)": "Balanced score (F1)",
                                  "ROC-AUC (macro, OvR)": "Ranking quality (AUC)"})
        pl = pl.melt(id_vars="Model", value_vars=["Accuracy", "Catch rate (recall)", "Balanced score (F1)",
                                                   "Ranking quality (AUC)"], var_name="Score", value_name="Value")
        show(style(px.bar(pl, x="Model", y="Value", color="Score", barmode="group",
                          color_discrete_sequence=[CYAN, ORANGE, GREEN, GREY], title="Model comparison on test data")
                   .update_yaxes(range=[0, 1], tickformat=".0%", title="Score")), left)

        model_pick = right.selectbox("Model for the two charts below", perf["Model"].tolist(),
                                     index=perf["Model"].tolist().index(best))
        m = cm[cm["Model"] == model_pick].pivot(index="Actual_Severity", columns="Predicted_Severity", values="Count")
        m_pct = m.div(m.sum(axis=1), axis=0)
        m_pct.index = [SEV_LABEL[i] for i in m_pct.index]
        m_pct.columns = [SEV_LABEL[i] for i in m_pct.columns]
        show(style(px.imshow(m_pct, text_auto=".0%", color_continuous_scale=ACCENT_SCALE, zmin=0, zmax=1,
                             labels=dict(x="What the model predicted", y="What really happened", color="Share"),
                             title="Predicted vs what really happened (each row adds up to 100%)"), 380), right)

        left, right = st.columns(2)
        r = pc[pc["Model"] == model_pick].copy()
        r["Severity Label"] = r["Severity"].map(SEV_LABEL)
        r["Catch rate"] = pd.cut(r["Recall"], [-1, 0.5, 0.7, 2],
                                 labels=["Low (under 50%)", "Medium (50-70%)", "Good (over 70%)"]).astype(str)
        show(style(px.bar(r, x="Recall", y="Severity Label", orientation="h", color="Catch rate",
                          color_discrete_map={"Good (over 70%)": GREEN, "Medium (50-70%)": ORANGE, "Low (under 50%)": RED},
                          category_orders={"Severity Label": SEV_ORDER[::-1]},
                          title="How many real cases were caught, per severity")
                   .update_xaxes(range=[0, 1], tickformat=".0%", title="Catch rate (recall)")), left)

        a_ = p["Actual_Severity"].value_counts().rename("Actual")
        b_ = p["Best_Model_Predicted"].value_counts().rename("Predicted")
        ap = pd.concat([a_, b_], axis=1).fillna(0).reindex([1, 2, 3, 4]).fillna(0)
        ap.index = [SEV_LABEL[i] for i in ap.index]
        ap = ap.reset_index().melt(id_vars="index", var_name="Type", value_name="Accidents")
        show(style(px.bar(ap, x="index", y="Accidents", color="Type", barmode="group",
                          color_discrete_map={"Actual": GREY, "Predicted": CYAN},
                          labels={"index": "Severity"}, title="Actual vs predicted accidents per severity")), right)

        left, right = st.columns(2)
        state_acc = p.groupby("State")["Best_Model_Correct"].mean().reset_index()
        show(style(px.choropleth(state_acc, locations="State", locationmode="USA-states", color="Best_Model_Correct",
                                 scope="usa", color_continuous_scale=ACCENT_SCALE, range_color=(0, 1),
                                 labels={"Best_Model_Correct": "Accuracy"}, title="Accuracy by state")), left)

        hr = p.groupby("Hour")["Best_Model_Correct"].mean().reset_index()
        show(style(px.line(hr, x="Hour", y="Best_Model_Correct", markers=True, color_discrete_sequence=[CYAN],
                           title="Accuracy by hour of day")
                   .update_yaxes(range=[0, 1], tickformat=".0%", title="Accuracy")
                   .update_xaxes(title="Hour of day (0 = midnight)")), right)

        st.markdown("### Details per severity level")
        detail = pc[pc["Model"] == model_pick].copy()
        detail["Severity"] = detail["Severity"].map(SEV_LABEL)
        detail = detail.set_index("Severity").drop(columns="Model")
        detail = detail.rename(columns={"Recall": "Recall (catch rate)", "F1-score": "F1", "Support": "Real cases in test data"})
        html_table(detail, {"Precision": "{:.1%}", "Recall (catch rate)": "{:.1%}", "F1": "{:.1%}",
                            "Real cases in test data": "{:,.0f}"})
        st.caption("The model does best on common severity levels. Rare levels, especially Severity 4, are harder "
                   "to catch. The test set is 20% of the data and was never used in training.")


# ================================================================= new pages: predict severity + accident forecast
MODEL_FILE, SETTINGS_FILE, MONTHLY_FILE = "best_model.joblib", "model_settings.json", "Monthly_Accident_Counts.csv"
SEV_COLS = [f"Severity_{k}" for k in (1, 2, 3, 4)]

ROAD_LABEL = {"Amenity": "Shop, cafe or other amenity nearby", "Crossing": "Pedestrian crossing", "Junction": "Junction",
              "Railway": "Railway crossing", "Station": "Station", "Stop": "Stop sign", "Traffic_Signal": "Traffic signal",
              "Bump": "Speed bump", "Give_Way": "Give-way sign", "Roundabout": "Roundabout",
              "Traffic_Calming": "Traffic calming", "Turning_Loop": "Turning loop"}
# One simple choice sets the four day/night columns of the dataset
LIGHT_COLS = ("Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight")
LIGHT_CHOICES = {"Day": ("Day", "Day", "Day", "Day"),
                 "Twilight (dusk or dawn)": ("Night", "Day", "Day", "Day"),
                 "Night": ("Night", "Night", "Night", "Night")}


def missing_files(names):
    return [n for n in names if not (Path(folder) / n).exists()]


def need_files(names, what):
    callout(f"<b>{what} needs {len(names)} more file(s)</b> in the data folder: "
            + ", ".join(f"<code>{html.escape(n)}</code>" for n in names)
            + ". They are created by the notebook.", warn=True)


def _patch_old_imputers(obj, _seen=None):
    """A model saved with an older scikit-learn lacks SimpleImputer._fill_dtype, which newer versions need.
    Re-create it from the fitted statistics so the old file still works. (Best fix: re-export the model
    from the notebook with the same scikit-learn version that runs the dashboard.)"""
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    if type(obj).__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype") and hasattr(obj, "statistics_"):
        obj._fill_dtype = obj.statistics_.dtype
    kids = []
    for attr in ("steps", "transformers_", "transformers"):
        for item in getattr(obj, attr, None) or []:
            kids.extend(x for x in (item[1:] if isinstance(item, tuple) else (item,)) if hasattr(x, "__dict__"))
    kids.extend(v for v in getattr(obj, "__dict__", {}).values() if hasattr(v, "fit") or hasattr(v, "steps"))
    for k in kids:
        _patch_old_imputers(k, _seen)


@st.cache_resource(show_spinner=False)
def load_model(folder):
    import joblib
    pipe = joblib.load(Path(folder) / MODEL_FILE)
    _patch_old_imputers(pipe)
    settings = json.loads((Path(folder) / SETTINGS_FILE).read_text())
    return pipe, settings


def model_categories(pipe, settings):
    """The category lists the model learned (State, Weather_Condition, light columns)."""
    ord_enc = pipe.named_steps["encode"].named_transformers_["cat"].named_steps["ord"]
    return {c: [str(v) for v in cats] for c, cats in zip(settings["categorical"], ord_enc.categories_)}


def match_label(value, options):
    """'Day' / 'Night' written the way the dataset spells it."""
    return next((o for o in options if o.lower() == value.lower()), value)


def score_bars(classes, proba, base, top):
    """Animated horizontal bars: the bar grows from 0; the thin mark shows how common the level is in the data."""
    rows = []
    for i, (c, p) in enumerate(zip(classes, proba)):
        rows.append(
            f'<div class="bar-row{" pick" if c == top else ""}"><div class="bar-top"><span>'
            f'<span class="name">{SEV_LABEL[c]}</span><span class="desc">{SEV_PLAIN[c]}</span></span>'
            f'<span class="val">{p:.0%}</span></div><div class="track">'
            f'<div class="fill" style="--w:{p * 100:.1f}%;--d:{i * 0.12:.2f}s;background:{SEV_COLOR[SEV_LABEL[c]]}"></div>'
            f'<div class="mark" style="left:{base[c] * 100:.1f}%" title="How common this level is in the data"></div>'
            f'</div></div>')
    st.markdown('<div class="bars">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def predictor():
    with keyed("page_predict"):
        hero("NEW ACCIDENT", "Predict the severity of a new accident",
             "Describe an accident: where, when, the weather and the road. The best model, trained on thousands of "
             "past accidents, shows which severity level the details point to.")
        sev_legend()

        gone = missing_files([MODEL_FILE, SETTINGS_FILE])
        if gone:
            need_files(gone, "This page")
            return
        try:
            with loading("Loading the trained model…", "Waking up the prediction engine"):
                pipe, settings = load_model(folder)
                cats = model_categories(pipe, settings)
        except Exception as e:      # usually a library-version mismatch with the notebook's environment
            st.error("The saved model could not be loaded. Check that scikit-learn, imbalanced-learn and lightgbm "
                     "have the same versions as in the notebook. "
                     f"Details: {type(e).__name__}: {e}")
            return

        model_name = settings.get("best_model", best_model)
        states = sorted(cats.get("State", []))
        wx_freq = acc["Weather_Condition"].value_counts()
        wx_opts = sorted(cats.get("Weather_Condition", []), key=lambda w: (-wx_freq.get(w, 0), w))
        top_state = acc["State"].value_counts().idxmax()

        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown("#### 1. Describe the accident")
            state = st.selectbox("State", states, index=states.index(top_state) if top_state in states else 0)
            when = st.date_input("Date", value=date.today())
            hour = st.slider("Hour of day", 0, 23, 17, help="0 = midnight, 17 = 5 pm")
            light = st.radio("Light", list(LIGHT_CHOICES), horizontal=True)
            wx = st.selectbox("Weather", wx_opts, index=wx_opts.index("Clear") if "Clear" in wx_opts else 0)
            c1, c2 = st.columns(2)
            temp = c1.number_input("Temperature (°F)", -40.0, 130.0, 62.0, step=1.0)
            hum = c2.number_input("Humidity (%)", 0.0, 100.0, 65.0, step=1.0)
            vis = c1.number_input("Visibility (miles)", 0.0, 100.0, 10.0, step=0.5)
            wind = c2.number_input("Wind speed (mph)", 0.0, 150.0, 7.0, step=1.0)
            pres = st.number_input("Pressure (inches)", 20.0, 32.5, 29.9, step=0.1)
            with st.expander("Road features at the location (optional)"):
                road_feats = [b for b in settings["binary"] if b not in ("Weekend", "Peak_Hour")]
                road = {b: int(st.checkbox(ROAD_LABEL.get(b, b), key=f"road_{b}")) for b in road_feats}

        # ---- build the input row exactly like the training data (the pipeline was fitted on all candidate columns)
        dow = when.weekday()
        row = {"Temperature(F)": temp, "Humidity(%)": hum, "Pressure(in)": pres, "Visibility(mi)": vis,
               "Wind_Speed(mph)": wind, "Hour": hour, "Day": when.day, "Month": when.month, "Day_of_Week": dow,
               "Weekend": int(dow >= 5), "Peak_Hour": int(hour in RUSH_HOURS),
               "Weather_Condition": wx, "State": state, **road}
        for col, val in zip(LIGHT_COLS, LIGHT_CHOICES[light]):
            row[col] = match_label(val, cats.get(col, []))
        cols = list(getattr(pipe.named_steps["encode"], "feature_names_in_", settings["selected_features"]))
        for c in cols:                      # attributes the model does not use are just placeholders
            row.setdefault(c, 0)
        X = pd.DataFrame([row])[cols]

        proba = pipe.predict_proba(X)[0]
        classes = [int(c) for c in pipe.named_steps["model"].classes_]
        top = classes[int(proba.argmax())]
        base = acc["Severity"].value_counts(normalize=True).reindex(classes).fillna(0)
        hi_score = float(sum(p for c, p in zip(classes, proba) if c >= 3))
        base34 = float(base[[c for c in classes if c >= 3]].sum())

        with right:
            st.markdown("#### 2. What the model expects")
            a, b = st.columns(2)
            insight(a, "Most likely level", SEV_LABEL[top], f"{SEV_PLAIN[top].capitalize()} · score {pct(proba.max())}")
            insight(b, "Leaning to Severity 3–4", pct(hi_score),
                    f"{'Above' if hi_score > base34 else 'Below'} the usual {pct(base34)}")
            score_bars(classes, proba, base, top)
            st.caption("Bar = the model's score. Thin mark = how common that level is in the data.")

            r4 = pc[(pc["Model"] == best_model) & (pc["Severity"] == 4)]["Recall"]
            r4_txt = f"; on the test data only {pct(r4.iloc[0])} of real ones were caught" if len(r4) else ""
            callout("<b>Read with care:</b> these are scores, not real-world chances (the model learned from re-balanced data). "
                    f"Severity 4 is rare and the hardest to spot{r4_txt}.", warn=True)
        st.caption(f"Model: {model_name}")

# ---------------------------------------------------------------- forecast
@st.cache_data
def load_monthly(folder):
    m = pd.read_csv(Path(folder) / MONTHLY_FILE)
    m["Date"] = pd.to_datetime(dict(year=m["Year"], month=m["Month"], day=1))
    m = m.sort_values("Date").set_index("Date")[SEV_COLS]
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="MS")).fillna(0)   # any missing month = 0
    m["Total"] = m[SEV_COLS].sum(axis=1)
    return m[m["Total"].cumsum() > 0]                                                 # drop empty months at the start


@st.cache_data(show_spinner=False)
def hw_forecast(values, start, steps, reps=1000):
    """Holt-Winters (damped trend, yearly seasonality) on log counts. Returns the point forecast and simulated paths."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    y = pd.Series(np.log1p(np.asarray(values, dtype=float)), index=pd.date_range(start, periods=len(values), freq="MS"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ETSModel(y, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit(disp=False)
        point = np.expm1(res.forecast(steps).to_numpy())
        seed = ({"rng": np.random.default_rng(42)} if "rng" in inspect.signature(res.simulate).parameters
                else {"random_state": 42})              # the argument name differs between statsmodels versions
        sims = np.expm1(np.asarray(res.simulate(steps, anchor="end", repetitions=reps, **seed)))
    return {"index": pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS"),
            "point": np.clip(point, 0, None), "sims": np.clip(sims, 0, None)}


def band(fc):
    return np.percentile(fc["sims"], 10, axis=1), np.percentile(fc["sims"], 90, axis=1)


def forecast_figure(hist, fc, title, naive=None, actual=None):
    lo, hi = band(fc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Reported", line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=fc["index"], y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["index"], y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(245,158,11,.25)", name="Likely range (8 in 10)"))
    x0 = [hist.index[-1]] if actual is None else []
    y0 = [hist.iloc[-1]] if actual is None else []
    fig.add_trace(go.Scatter(x=x0 + list(fc["index"]), y=y0 + list(fc["point"]), mode="lines", name="Forecast",
                             line=dict(color=ORANGE, width=2.5, dash="dash")))
    if naive is not None:
        fig.add_trace(go.Scatter(x=fc["index"], y=naive, mode="lines", name="Same month last year",
                                 line=dict(color=GREY, width=2, dash="dot")))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines+markers", name="Reported (hidden from the model)",
                                 line=dict(color=ACCENT, width=2.5)))
    fig.update_layout(title=title)
    fig.update_yaxes(title="Accidents per month", rangemode="tozero")
    return style(fig, 400)


def forecaster():
    with keyed("page_forecast"):
        hero("FORECAST", "How many accidents might be reported next year?",
             "A forecasting model studies the monthly pattern of the whole dataset (millions of records) and carries it "
             "forward. It is a separate model from the severity predictor.")

        gone = missing_files([MONTHLY_FILE])
        if gone:
            need_files(gone, "This page")
            return
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            st.error("This page needs the 'statsmodels' package. Run:  pip install statsmodels")
            return

        m = load_monthly(folder)
        last = m.index.max()
        if len(m) < 36:
            st.warning("The monthly file has fewer than 36 months, which is too short for a seasonal forecast.")
            return

        callout(f"<b>Read with care:</b> the data ends in <b>{last:%B %Y}</b>, so later years are forecasts that cannot be "
                "checked yet. Counts are accidents <i>reported in this dataset</i> (more sources over time), not the true national total.",
                warn=True)

        # ---- controls
        first_year = last.year if last.month < 12 else last.year + 1
        c1, c2 = st.columns(2)
        year = c1.selectbox("Which year?", list(range(first_year, last.year + 4)))
        mode = c2.radio("Split by severity using the mix of", ["Latest 12 months", "Latest 3 years", "All years"], horizontal=True,
                        help="The forecast total is shared between severity levels using the share each level had in this period.")
        n_mix = {"Latest 12 months": 12, "Latest 3 years": 36, "All years": len(m)}[mode]

        steps = (year - last.year) * 12 + (12 - last.month)
        with loading("Fitting the forecast model…", "Learning the monthly pattern"):
            fc = hw_forecast(tuple(m["Total"].astype(float)), str(m.index[0].date()), steps)

        # ---- the year's total = months already reported + simulated months
        in_year = fc["index"].year == year
        have = m[m.index.year == year]
        rem = fc["sims"][in_year].sum(axis=0)                     # simulated total of the months still to come
        r10, r50, r90 = np.percentile(rem, [10, 50, 90])
        a_tot = float(have["Total"].sum())
        t10, t50, t90 = a_tot + r10, a_tot + r50, a_tot + r90

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Expected in {year}", f"{t50:,.0f}",
                  help="Middle estimate. " + (f"Includes {len(have)} month(s) already reported in the data." if len(have) else ""))
        k2.metric("Likely range (8 in 10)", f"{t10:,.0f} – {t90:,.0f}",
                  help="If the pattern continues, the real number should land inside this range about 8 times out of 10.")
        prev = m[m.index.year == year - 1]["Total"]
        if len(prev) == 12:
            k3.metric(f"Change vs {year - 1}", f"{(t50 / prev.sum() - 1) * 100:+.1f}%", help=f"{year - 1} reported: {prev.sum():,.0f}")
        else:
            k3.metric("Months forecast", f"{int(in_year.sum())} of 12")

        show(forecast_figure(m["Total"], fc, f"Accidents reported per month, with the forecast to {year}"))

        # ---- split by severity
        tail = m.tail(n_mix)[SEV_COLS].sum()
        share = tail / tail.sum()
        rows = []
        for k, col in zip((1, 2, 3, 4), SEV_COLS):
            seen = float(have[col].sum())
            rows.append({"Severity": f"{SEV_LABEL[k]} ({SEV_PLAIN[k]})", "Share": f"{share[col]:.1%}",
                         "Expected": f"{seen + share[col] * r50:,.0f}",
                         "Likely range": f"{seen + share[col] * r10:,.0f} – {seen + share[col] * r90:,.0f}"})
        sev_tbl = pd.DataFrame(rows).set_index("Severity")

        st.markdown(f"### {year} by severity level")
        html_table(sev_tbl)
        st.caption(f"Each level gets its share of the forecast total ({mode.lower()}); the mix is assumed to stay the same.")

        # ---- evidence: can we trust it?
        section("Can we trust the forecast?",
                "Test: hide one year from the model, forecast it, then compare with what was really reported.")
        years = sorted(set(m.index.year))
        ok_years = [y for y in years if (m.index.year == y).sum() == 12 and (m.index < f"{y}-01-01").sum() >= 36]
        if not ok_years:
            st.info("Not enough complete years to run this test.")
        else:
            by = st.selectbox("Year to hide and forecast", ok_years, index=len(ok_years) - 1)
            train, test = m[m.index < f"{by}-01-01"]["Total"], m[m.index.year == by]["Total"]
            with loading("Running the test…", f"Forecasting {by} from the years before it"):
                bt = hw_forecast(tuple(train.astype(float)), str(train.index[0].date()), len(test))
            naive = train.iloc[-12:].to_numpy()                   # same month, one year earlier
            lo, hi = band(bt)
            act = test.to_numpy()

            def scores(f):
                return {"Typical monthly miss": float(np.mean(np.abs(act - f) / act) * 100),
                        "Typical miss (accidents)": float(np.sqrt(np.mean((act - f) ** 2))),
                        "Miss on the year's total": float((f.sum() / act.sum() - 1) * 100)}
            tbl = pd.DataFrame({"Holt-Winters forecast": scores(bt["point"]), "Same month last year": scores(naive)}).T
            tbl.index.name = "Method"
            hw, nv = tbl.loc["Holt-Winters forecast"], tbl.loc["Same month last year"]
            cover = float(np.mean((act >= lo) & (act <= hi)))
            tot_key = "Miss on the year's total"
            better = hw["Typical monthly miss"] < nv["Typical monthly miss"]

            callout(f"<b>In plain words:</b> pretending {by} had not happened, the forecast was off by about "
                    f"<b>{hw['Typical monthly miss']:.0f}%</b> in a typical month, and its total for the year was "
                    f"<b>{abs(hw[tot_key]):.1f}% too "
                    f"{'high' if hw[tot_key] > 0 else 'low'}</b>, and {cover:.0%} of the months landed inside the likely range. "
                    + (f"That beats simply repeating the same month from the year before ({nv['Typical monthly miss']:.0f}% off)."
                       if better else
                       f"That is <u>not</u> better than simply repeating the same month from the year before "
                       f"({nv['Typical monthly miss']:.0f}% off), so treat this forecast with extra caution."),
                    warn=not better)

            show(forecast_figure(train, bt, f"The test on {by}: forecast made from the earlier years vs what was reported",
                                 naive=naive, actual=test))
            html_table(tbl, {"Typical monthly miss": "{:.1f}%", "Typical miss (accidents)": "{:,.0f}",
                             "Miss on the year's total": "{:+.1f}%"})
            st.caption("“Typical miss (accidents)” is RMSE, in accidents per month. “Same month last year” is a simple "
                       "benchmark: the forecast is only useful if it does better.")

        with st.expander("ℹ️ How this forecast is made"):
            st.markdown(
                "- **Data:** monthly counts from the **full** accident dataset (not the 60,000-record sample), "
                "after removing duplicate reports.\n"
                "- **Model:** Holt-Winters exponential smoothing with a damped trend and yearly seasonality, fitted on "
                "log counts so forecasts stay positive.\n"
                "- **Likely range:** 1,000 simulated futures from the fitted model; the range runs from the 10th to the 90th percentile.\n"
                "- **Severity split:** the forecast total times the chosen historical share of each severity level.\n"
                "- **Limits:** reporting coverage changed over time, so past growth may not continue, and the further ahead "
                "you look, the wider the real uncertainty.")


# ================================================================= navigation (real top menu)
brand_bar()
PAGES = [
    st.Page(overview, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(explorer, title="Risk Explorer", icon="🔎", url_path="explore"),
    st.Page(training, title="Model Training", icon="🧠", url_path="training"),
    st.Page(results, title="Test Results", icon="🎯", url_path="results"),
    st.Page(predictor, title="Predict Severity", icon="🔮", url_path="predict"),
    st.Page(forecaster, title="Accident Forecast", icon="📈", url_path="forecast"),
]
try:
    nav = st.navigation(PAGES, position="top")
except TypeError:  # Streamlit 1.36 - 1.45: menu goes to the sidebar (hamburger button on phones)
    nav = st.navigation(PAGES)
nav.run()

st.markdown("---")
st.caption("Data: US Accidents (Kaggle), random sample · Severity = traffic-disruption class (1–4), not injury level · "
           "3 machine-learning models compared, Holt-Winters for the forecast · Python, scikit-learn, LightGBM, statsmodels, Streamlit")
data_settings()
