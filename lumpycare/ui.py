"""Shared LumpyCare visual system for the Streamlit application."""
from __future__ import annotations
import streamlit as st

def inject_theme(dark_mode: bool = False) -> None:
    palette = {
        "page": "#101a1b" if dark_mode else "#f3f7f5",
        "surface": "#172526" if dark_mode else "#ffffff",
        "ink": "#e7f2ed" if dark_mode else "#173b30",
        "muted": "#a5bbb1" if dark_mode else "#687d73",
        "line": "#314747" if dark_mode else "#dce8e1",
        "green": "#35b58a" if dark_mode else "#147a5b",
        "green_soft": "#193d34" if dark_mode else "#e0f2e9",
        "shadow": "0 14px 36px rgba(0,0,0,.18)" if dark_mode else "0 12px 30px rgba(26,69,52,.06)",
    }
    css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --lc-page:%(page)s; --lc-surface:%(surface)s; --lc-ink:%(ink)s; --lc-muted:%(muted)s;
      --lc-line:%(line)s; --lc-green:%(green)s; --lc-green-soft:%(green_soft)s; --lc-shadow:%(shadow)s; }
    html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
    html,body,[class*="css"] { color:var(--lc-ink) !important; }
    .stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown span,label,[data-testid="stWidgetLabel"] p { color:var(--lc-ink); }
    .stCaption,[data-testid="stCaptionContainer"] p,.muted,.lc-subtitle { color:var(--lc-muted) !important; }
    input,textarea,[data-baseweb="select"]>div { color:var(--lc-ink) !important; background:var(--lc-surface) !important; border-color:var(--lc-line) !important; }
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--lc-ink) !important; }
    .stApp { background:var(--lc-page); color:var(--lc-ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:var(--lc-surface); border-right:1px solid var(--lc-line); }
    .lc-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .35rem 1rem; }
    .lc-brand-mark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px;
      background:linear-gradient(145deg,#0d6c52,#2bb58a); color:white; font-size:1.35rem; }
    .lc-brand-name { color:var(--lc-ink); font:800 1.2rem 'Manrope',sans-serif; letter-spacing:-.04em; }
    .lc-brand-sub { color:var(--lc-muted); font-size:.7rem; }
    .lc-topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.8rem 1rem;
      margin-bottom:1.15rem; background:var(--lc-surface); border:1px solid var(--lc-line);
      border-radius:16px; box-shadow:var(--lc-shadow); }
    .lc-eyebrow { color:var(--lc-green); text-transform:uppercase; letter-spacing:.13em; font-size:.68rem; font-weight:800; }
    .lc-page-title { color:var(--lc-ink); font:800 clamp(1.5rem,3vw,2rem) 'Manrope',sans-serif; letter-spacing:-.045em; }
    .lc-subtitle,.muted { color:var(--lc-muted); font-size:.9rem; }
    .hero { padding:clamp(1.35rem,3vw,2.25rem); border-radius:22px; color:#fff;
      background:radial-gradient(circle at 85% 15%,rgba(105,220,174,.24),transparent 28%),linear-gradient(120deg,#103b32,#147a5b 62%%,#269a70);
      box-shadow:0 16px 38px rgba(15,83,60,.17); margin-bottom:1.25rem; }
    .hero .eyebrow { color:#bcebd4; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:800; }
    .hero h1 { color:#fff; font:800 clamp(1.8rem,4vw,2.55rem) 'Manrope',sans-serif; letter-spacing:-.045em; margin:.3rem 0; }
    .hero p { color:#e0f3e9; margin:0; max-width:680px; }
    .metric-card,.panel { background:var(--lc-surface); border:1px solid var(--lc-line); border-radius:18px;
      padding:1.05rem 1.15rem; box-shadow:var(--lc-shadow); }
    .metric-card { min-height:106px; transition:transform .18s ease,border-color .18s ease; }
    .metric-card:hover,.panel:hover { transform:translateY(-2px); border-color:var(--lc-green); }
    .metric-label { color:var(--lc-muted); font-size:.8rem; font-weight:700; }
    .metric-value { color:var(--lc-ink); font:800 1.7rem 'Manrope',sans-serif; letter-spacing:-.04em; margin-top:.25rem; }
    div[data-testid="stFileUploader"],div[data-testid="stCameraInput"] { background:var(--lc-surface); border:1px dashed #8eb8a3; border-radius:15px; padding:.7rem; }
    .stButton>button,.stDownloadButton>button { border-radius:11px; font-weight:700; min-height:2.65rem; border:1px solid var(--lc-line); }
    .stButton>button[kind="primary"] { background:var(--lc-green); border-color:var(--lc-green); }
    [data-testid="stMetric"] { background:var(--lc-surface); border:1px solid var(--lc-line); padding:1rem; border-radius:15px; }
    [data-testid="stDataFrame"] { border:1px solid var(--lc-line); border-radius:14px; overflow:hidden; }
    button:focus-visible,input:focus-visible,[tabindex="0"]:focus-visible { outline:3px solid #55c99b !important; outline-offset:2px !important; }
    @media(max-width:760px) { .hero{border-radius:17px}.metric-card{min-height:88px}.lc-topbar{padding:.7rem;border-radius:13px} section.main>div{padding-left:1rem;padding-right:1rem} }
    @media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important;transition-duration:.01ms !important; } }
    """
    for key, value in palette.items():
        css = css.replace("%(" + key + ")s", value)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="lc-brand"><div class="lc-brand-mark">🐄</div><div><div class="lc-brand-name">LumpyCare</div><div class="lc-brand-sub">VETERINARY AI WORKSPACE</div></div></div>', unsafe_allow_html=True)
        st.caption("WORKSPACE")
        page = st.radio("Main navigation", ["Dashboard","New Scan","Care & First Aid","Hospitals & Doctors","Appointments","Doctor Portal","History","Cattle Profiles","Outbreak Map","Reports","Model Insights","Settings"], label_visibility="collapsed")
        st.divider()
        st.caption("APPEARANCE")
        st.toggle("Dark mode", key="lc_dark_mode", help="Switch the current session to a dark color theme.")
        st.divider()
        st.caption("Research prototype · Vet confirmation required")
        st.caption("Streamlit · FastAPI · OpenCV · scikit-learn")
    return page

def render_topbar(page: str) -> None:
    st.markdown(
        f'<div class="lc-topbar"><div><div class="lc-eyebrow">LumpyCare / Workspace</div><div class="lc-page-title">{page}</div><div class="lc-subtitle">AI-assisted cattle skin screening and herd awareness</div></div><div style="text-align:right"><span style="color:var(--lc-green);font-weight:700;font-size:.8rem">● LOCAL SESSION</span><div class="muted">Prototype environment</div></div></div>',
        unsafe_allow_html=True,
    )
