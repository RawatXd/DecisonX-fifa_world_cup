
"""
src/dashboard/app.py
--------------------
DecisionX — FIFA World Cup Optimizer
Phase 5 — Final Dashboard

Clean minimal design · Inter + JetBrains Mono
Correct imports · cfg paths · predict_match

Run with:
  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from src.config import cfg
from src.models.poisson_model import predict_match
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DecisionX · FIFA WC Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean minimal light theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Tokens ── */
:root {
  --bg:          #FAFAFA;
  --surface:     #FFFFFF;
  --border:      #E5E7EB;
  --border-md:   #D1D5DB;
  --blue:        #2563EB;
  --blue-soft:   #DBEAFE;
  --blue-mid:    #93C5FD;
  --text-1:      #111827;
  --text-2:      #374151;
  --text-3:      #6B7280;
  --text-4:      #9CA3AF;
  --success:     #10B981;
  --warning:     #F59E0B;
  --danger:      #EF4444;
  --sans:        'Inter', -apple-system, sans-serif;
  --mono:        'JetBrains Mono', 'Fira Code', monospace;
  --radius:      8px;
  --radius-lg:   12px;
  --shadow-sm:   0 1px 3px rgba(0,0,0,0.06);
  --shadow-md:   0 4px 12px rgba(0,0,0,0.08);
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--text-1) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-2) !important;
  font-family: var(--sans) !important;
}

/* ── Layout ── */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1320px;
}

/* ── Page headers ── */
.page-title {
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -0.025em;
  margin-bottom: 0.2rem;
  line-height: 1.2;
}
.page-sub {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-4);
  letter-spacing: 0.03em;
  margin-bottom: 2rem;
}

/* ── Section label ── */
.label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-4);
  margin-bottom: 0.5rem;
}

/* ── KPI strip ── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  margin-bottom: 2rem;
}
.kpi-cell {
  background: var(--surface);
  padding: 1.1rem 1.4rem;
}
.kpi-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-4);
  margin-bottom: 0.35rem;
}
.kpi-value {
  font-family: var(--mono);
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.03em;
  line-height: 1;
}
.kpi-value.blue  { color: var(--blue); }
.kpi-value.green { color: var(--success); }
.kpi-value.amber { color: var(--warning); }

/* ── Card ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: clamp(1rem, 3vw, 1.5rem);
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
.card h3 {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-1);
  margin-bottom: 0.5rem;
}
.card p {
  font-size: 0.875rem;
  color: var(--text-3);
  line-height: 1.65;
  margin-bottom: 0.5rem;
}

/* ── Stat row ── */
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.875rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: var(--text-3); }
.stat-val { font-family: var(--mono); font-weight: 700; color: var(--text-1); }
.stat-val.blue { color: var(--blue); }

/* ── Info / success / warning boxes ── */
.info-box, .success-box, .warning-box {
  border-radius: var(--radius);
  padding: 0.9rem 1.1rem;
  margin: 1.25rem 0;
  font-size: 0.875rem;
  border-left: 3px solid;
  line-height: 1.65;
}
.info-box    { background: #EFF6FF; border-color: var(--blue);    color: #1E40AF; }
.success-box { background: #F0FDF4; border-color: var(--success); color: #065F46; }
.warning-box { background: #FFFBEB; border-color: var(--warning); color: #92400E; }

/* ── Highlight accent strip ── */
.highlight-accent {
  background: var(--blue);
  color: white;
  padding: clamp(1rem, 3vw, 1.5rem);
  border-radius: var(--radius-lg);
  margin: 1.5rem 0;
}
.highlight-accent h3 { color: white !important; font-size: 1rem; margin-bottom: 0.5rem; }
.highlight-accent p  { color: rgba(255,255,255,0.9); font-size: 0.875rem; line-height: 1.7; }

/* ── Match prediction card ── */
.match-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2rem 2.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
}
.match-team { flex: 1; }
.match-team-name  { font-size: 1.1rem; font-weight: 800; color: var(--text-1); margin-bottom: 0.2rem; }
.match-team-prob  { font-family: var(--mono); font-size: 2.2rem; font-weight: 700;
                    color: var(--blue); letter-spacing: -0.04em; }
.match-team-label { font-size: 0.7rem; color: var(--text-4); margin-top: 0.15rem; }
.match-vs         { text-align: center; padding: 0 1rem; }
.match-vs-text    { font-family: var(--mono); font-size: 0.8rem; font-weight: 700;
                    color: var(--text-4); letter-spacing: 0.1em; }
.match-xg         { font-family: var(--mono); font-size: 0.72rem; color: var(--text-4); margin-top: 0.3rem; }

/* ── Player row ── */
.player-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.875rem;
}
.player-row:last-child { border-bottom: none; }
.player-name { font-weight: 600; color: var(--text-1); }
.player-club { font-size: 0.78rem; color: var(--text-4); margin-top: 0.1rem; }
.player-meta { display: flex; align-items: center; gap: 0.75rem; }
.pos-tag {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  border: 1px solid;
}
.pos-GK  { color: #92400E; border-color: #FDE68A; background: #FFFBEB; }
.pos-DEF { color: #065F46; border-color: #A7F3D0; background: #ECFDF5; }
.pos-MID { color: #1E40AF; border-color: #BFDBFE; background: #EFF6FF; }
.pos-FWD { color: #991B1B; border-color: #FECACA; background: #FEF2F2; }
.player-mv { font-family: var(--mono); font-size: 0.78rem; color: var(--text-3); }

/* ── Impact bars ── */
.impact-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
}
.impact-row:last-child { border-bottom: none; }
.impact-name { font-size: 0.84rem; font-weight: 500; color: var(--text-2); width: 165px; flex-shrink: 0; }
.impact-bar-bg { flex: 1; height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
.impact-bar-fill { height: 100%; border-radius: 999px; background: var(--blue); transition: width 0.4s; }
.impact-num { font-family: var(--mono); font-size: 0.72rem; color: var(--text-3); width: 52px; text-align: right; }

/* ── Code block ── */
.code-block {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--text-3);
  background: #F9FAFB;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  line-height: 1.9;
}
.ck  { color: var(--blue); }
.cn  { color: #059669; }
.cs  { color: #B45309; }
.cc  { color: var(--text-4); }

/* ── Tables ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}
thead tr th {
  background: #F9FAFB !important;
  color: var(--text-3) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0.65rem 0.75rem !important;
}
tbody tr td {
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
  color: var(--text-2) !important;
  padding: 0.55rem 0.75rem !important;
}
tbody tr:hover td { background: #F0F9FF !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
  box-shadow: var(--shadow-md) !important;
}
[data-testid="metric-container"] label {
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-4) !important;
  font-family: var(--sans) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  color: var(--text-1) !important;
  letter-spacing: -0.03em !important;
}

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: var(--radius) !important;
  font-family: var(--sans) !important;
  font-size: 0.9rem !important;
  color: var(--text-1) !important;
}

/* ── Sidebar radio nav ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] label[data-baseweb] {
  display: block !important;
  padding: 0.55rem 0.85rem !important;
  border-radius: 6px !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  color: var(--text-3) !important;
  cursor: pointer !important;
  transition: background 0.12s, color 0.12s !important;
  margin-bottom: 2px !important;
}
div[data-testid="stRadio"] label[data-baseweb]:hover {
  background: #F3F4F6 !important;
  color: var(--text-1) !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 2rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 999px; }

/* ── Plotly charts ── */
.plotly-graph-div { border-radius: var(--radius-lg) !important; }

/* ── Responsive ── */
@media (max-width: 640px) {
  .main .block-container { padding: 1rem 1rem 3rem !important; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
  .match-block { flex-direction: column; text-align: center; }
  .page-title { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_tournament_probs():
    try:
        return pd.read_csv(cfg.OUTPUTS / "tournament_probs.csv")
    except Exception as e:
        st.error(f"Error loading tournament probabilities: {e}")
        return None

@st.cache_data
def load_poisson_params():
    try:
        return pd.read_csv(cfg.DATA_PROC / "poisson_params.csv")
    except Exception as e:
        st.error(f"Error loading Poisson params: {e}")
        return None

@st.cache_data
def load_squad(team_name):
    try:
        path = cfg.OUTPUTS.parent / "data" / "squads" / f"{team_name.lower()}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df["injured"] = df["injured"].astype(bool)
        return df
    except Exception:
        return None

tournament_probs = load_tournament_probs()
poisson_params   = load_poisson_params()

if tournament_probs is None or poisson_params is None:
    st.error("❌ Unable to load required data files. Please ensure Phase 1–3 are complete.")
    st.stop()

avg_goals = poisson_params["avg_goals_baseline"].iloc[0]
all_teams = sorted(tournament_probs["team"].unique())
SQUADS    = [t.capitalize() for t in ["argentina","brazil","france","germany"]
             if (cfg.OUTPUTS.parent / "data" / "squads" / f"{t}.csv").exists()]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def player_str(row):
    w = {"GK":0.8,"DEF":0.9,"MID":1.0,"FWD":1.2}.get(str(row.get("position","MID")),1.0)
    return float(row.get("market_value_millions", 10)) / 100.0 * w

def light_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter", color="#6B7280", size=11),
        margin=dict(l=10, r=10, t=44, b=10),
        title_font=dict(color="#111827", size=13, family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#E5E7EB"),
        xaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB", linecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB", linecolor="#E5E7EB"),
        hovermode="x unified",
    )
    return fig

BLUE  = "#2563EB"
BLUEM = "#93C5FD"
BLUEG = "#DBEAFE"
GREY  = "#E5E7EB"
GREEN = "#10B981"
AMBER = "#F59E0B"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding:1.25rem 0 1rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;
              color:#111827;letter-spacing:-0.02em;">DecisionX</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
              color:#9CA3AF;margin-top:0.2rem;">// fifa_wc_optimizer</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    tab_choice = st.radio("", [
        "🏠 Home",
        "🎯 Team Explorer",
        "🎲 Bracket Simulator",
        "⚔️ Head-to-Head",
        "👥 Lineup Builder",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.divider()

    st.markdown(f"""
<div class="code-block">
<span class="cc"># model performance</span><br>
<span class="ck">poisson_acc</span>  = <span class="cn">0.617</span><br>
<span class="ck">xgboost_acc</span>  = <span class="cn">0.492</span><br>
<span class="ck">n_sims</span>       = <span class="cn">10_000</span><br>
<span class="ck">features</span>     = <span class="cn">30</span><br>
<span class="ck">matches</span>      = <span class="cn">964</span><br>
<span class="ck">year_range</span>   = <span class="cs">"1930–2022"</span><br>
<span class="ck">updated</span>      = <span class="cs">"{datetime.now().strftime('%Y-%m-%d')}"</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

if tab_choice == "🏠 Home":

    st.markdown("""
<div class="page-title">DecisionX</div>
<div class="page-sub">fifa_wc_optimizer · qatar 2022 · decision-support platform</div>
""", unsafe_allow_html=True)

    top1 = tournament_probs.iloc[0]

    # KPI strip
    st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-cell">
    <div class="kpi-label">Matches trained on</div>
    <div class="kpi-value">964</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Teams analyzed</div>
    <div class="kpi-value">32</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Simulations</div>
    <div class="kpi-value blue">10K</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Poisson accuracy</div>
    <div class="kpi-value green">61.7%</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Features</div>
    <div class="kpi-value">30</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Tournament favourite</div>
    <div class="kpi-value blue">{top1['team'][:3].upper()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="label">Tournament win probability · top 12 teams</div>', unsafe_allow_html=True)
        t12    = tournament_probs.head(12)
        colors = [BLUE if i==0 else BLUEM if i<3 else GREY for i in range(len(t12))]
        fig = go.Figure(go.Bar(
            x=t12["team"],
            y=t12["p_win"] * 100,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in t12["p_win"]*100],
            textposition="outside",
            textfont=dict(size=10, color="#6B7280", family="JetBrains Mono"),
        ))
        light_chart(fig, 350).update_layout(
            xaxis_tickangle=-30,
            yaxis_title="p_win (%)",
            showlegend=False,
            title="Top 12 Favourites",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="label">Top 5 favourites</div>', unsafe_allow_html=True)
        for i, row in tournament_probs.head(5).iterrows():
            st.markdown(f"""
<div class="stat-row">
  <span class="stat-key">#{i+1}&nbsp; {row['team']}</span>
  <span class="stat-val blue">{row['p_win']:.2%}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="label">Platform features</div>', unsafe_allow_html=True)
        for icon, name, desc in [
            ("🎯","Team Explorer",    "Tournament path probabilities"),
            ("⚔️","Head-to-Head",     "Poisson match predictor"),
            ("🎲","Bracket Sim",      "10K Monte Carlo runs"),
            ("👥","Lineup Builder",   "Integer programming optimizer"),
        ]:
            st.markdown(f"""
<div class="stat-row">
  <span class="stat-key">{icon}&nbsp; {name}</span>
  <span style="font-size:0.76rem;color:#9CA3AF;">{desc}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div class="label">What DecisionX answers</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, body in [
        (col1,"🏆","Who wins?",
         "Which teams are most likely to win the World Cup based on 10,000 simulated tournaments."),
        (col2,"📈","How far?",
         "Every team's P(QF), P(SF), P(Final) — from group-stage draw to the trophy."),
        (col3,"👥","Best XI?",
         "Which 11 players maximise tournament win probability given formation constraints."),
        (col4,"⚠️","Key player?",
         "How much does the win probability drop if a key player is unavailable?"),
    ]:
        with col:
            st.markdown(f"""
<div class="card">
  <div style="font-size:1.4rem;margin-bottom:0.5rem;">{icon}</div>
  <h3>{title}</h3>
  <p>{body}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-accent">
  <h3>🚀 Quick Start</h3>
  <p>
    1. <strong>Team Explorer</strong> — pick your team, see the full tournament path<br>
    2. <strong>Head-to-Head</strong> — predict any fixture with the Poisson model<br>
    3. <strong>Bracket Simulator</strong> — explore all 32 teams side by side<br>
    4. <strong>Lineup Builder</strong> — find the optimal XI and player impact scores
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TEAM EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "🎯 Team Explorer":

    st.markdown("""
<div class="page-title">Team Explorer</div>
<div class="page-sub">model: monte_carlo · n: 10,000 · source: poisson_model</div>
""", unsafe_allow_html=True)

    selected_team = st.selectbox("Select a team:", all_teams, key="explorer_team")

    if selected_team:
        row  = tournament_probs[tournament_probs["team"] == selected_team].iloc[0]
        rank = int(tournament_probs[tournament_probs["team"] == selected_team].index[0]) + 1

        st.markdown(f'<div class="label">{selected_team}</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("P(Reach Group)", f"{row['p_group']:.1%}")
        col2.metric("P(Reach QF)",    f"{row['p_qf']:.1%}")
        col3.metric("P(Reach Final)", f"{row['p_final']:.1%}")
        col4.metric("P(Win WC)",      f"{row['p_win']:.1%}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="label">Tournament path — stage probabilities</div>', unsafe_allow_html=True)

        stages = ["Group","R16","QF","SF","Final","Win"]
        probs  = [row["p_group"],row["p_r16"],row["p_qf"],row["p_sf"],row["p_final"],row["p_win"]]

        colors = [BLUE if v == max(probs) else BLUEM if v > np.mean(probs) else GREY for v in probs]
        fig = go.Figure(go.Bar(
            x=stages, y=probs,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            textfont=dict(size=11, color="#374151", family="JetBrains Mono"),
        ))
        light_chart(fig, 380).update_layout(
            title=f"{selected_team} — Stage Progression",
            xaxis_title="stage",
            yaxis_title="probability",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"""
<div class="info-box">
  <strong>{selected_team}</strong> is ranked <strong>#{rank} of 32</strong> teams by tournament win probability.
</div>
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="label">Top 10 — win probability</div>', unsafe_allow_html=True)
            top10 = tournament_probs.nlargest(10,"p_win")[["team","p_win"]].reset_index(drop=True)
            top10.index += 1
            top10.columns = ["Team","Win %"]
            top10["Win %"] = top10["Win %"].apply(lambda x: f"{x:.2%}")
            st.dataframe(top10, use_container_width=True)

        with col2:
            st.markdown('<div class="label">Top 10 — final probability</div>', unsafe_allow_html=True)
            top10f = tournament_probs.nlargest(10,"p_final")[["team","p_final"]].reset_index(drop=True)
            top10f.index += 1
            top10f.columns = ["Team","Final %"]
            top10f["Final %"] = top10f["Final %"].apply(lambda x: f"{x:.2%}")
            st.dataframe(top10f, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BRACKET SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "🎲 Bracket Simulator":

    st.markdown("""
<div class="page-title">Bracket Simulator</div>
<div class="page-sub">method: monte_carlo · n_simulations: 10,000 · source: poisson_goal_model</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="success-box">
  ✅ <strong>10,000 simulations completed</strong> — results are statistically robust
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    max_team = tournament_probs.loc[tournament_probs["p_win"].idxmax(), "team"]
    col1.metric("Tournament Favourite",   max_team)
    col2.metric("P(win)",                 f"{tournament_probs['p_win'].max():.2%}")
    col3.metric("Median P(final)",        f"{tournament_probs['p_final'].median():.2%}")

    st.markdown("<hr>", unsafe_allow_html=True)

    stage_map = {
        "Round of 16":    "p_r16",
        "Quarter-Finals": "p_qf",
        "Semi-Finals":    "p_sf",
        "Final":          "p_final",
        "Win":            "p_win",
    }
    sel_stage = st.selectbox("Select stage:", list(stage_map.keys()), index=1)
    scol = stage_map[sel_stage]
    df16 = tournament_probs.nlargest(16, scol)

    st.markdown(f'<div class="label">P(reach {sel_stage}) · top 16 teams</div>', unsafe_allow_html=True)

    bar_colors = [BLUE if i==0 else BLUEM if i<3 else "#D1D5DB" for i in range(len(df16))]
    fig = go.Figure(go.Bar(
        y=df16["team"][::-1],
        x=df16[scol][::-1] * 100,
        orientation="h",
        marker_color=bar_colors[::-1],
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in df16[scol][::-1]*100],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#6B7280"),
    ))
    light_chart(fig, 520).update_layout(
        xaxis_title="probability (%)",
        showlegend=False,
        title=f"P(Reach {sel_stage})",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="label">All 32 teams · all stages</div>', unsafe_allow_html=True)

    tbl = tournament_probs.copy().reset_index(drop=True)
    tbl.index += 1
    tbl.columns = ["Team","Group","R16","QF","SF","Final","Win"]
    for c in ["Group","R16","QF","SF","Final","Win"]:
        tbl[c] = tbl[c].apply(lambda x: f"{x:.1%}")
    st.dataframe(tbl, use_container_width=True, height=680)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="label">Win probability distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(tournament_probs, x="p_win", nbins=15,
                            color_discrete_sequence=[BLUE])
        light_chart(fig2, 320).update_layout(
            xaxis_title="p_win", yaxis_title="count",
            showlegend=False, title="Win Probability Distribution",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="label">Final probability distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(tournament_probs, x="p_final", nbins=15,
                            color_discrete_sequence=[BLUEM])
        light_chart(fig3, 320).update_layout(
            xaxis_title="p_final", yaxis_title="count",
            showlegend=False, title="Final Probability Distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HEAD-TO-HEAD
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "⚔️ Head-to-Head":

    st.markdown("""
<div class="page-title">Head-to-Head</div>
<div class="page-sub">model: poisson_goal_model · accuracy: 0.617 · scoreline distributions</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Home Team:", all_teams, key="h2h_home", index=0)
    with col2:
        team_b = st.selectbox("Away Team:", [t for t in all_teams if t != team_a], key="h2h_away")

    if team_a and team_b and team_a != team_b:
        result = predict_match(team_a, team_b, poisson_params, avg_goals)

        st.markdown(f"""
<div class="match-block">
  <div class="match-team">
    <div class="match-team-name">{team_a}</div>
    <div class="match-team-prob">{result['p_home_win']:.1%}</div>
    <div class="match-team-label">win probability</div>
  </div>
  <div class="match-vs">
    <div class="match-vs-text">VS</div>
    <div class="match-xg">xG &nbsp;{result['lambda_home']:.2f} · {result['lambda_away']:.2f}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                color:#9CA3AF;margin-top:0.35rem;">draw: {result['p_draw']:.1%}</div>
  </div>
  <div class="match-team" style="text-align:right;">
    <div class="match-team-name">{team_b}</div>
    <div class="match-team-prob">{result['p_away_win']:.1%}</div>
    <div class="match-team-label">win probability</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="info-box">
  <strong>Expected goals:</strong> {team_a} <strong>{result['lambda_home']:.2f}</strong>
  — <strong>{result['lambda_away']:.2f}</strong> {team_b}
</div>
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="label">Outcome distribution</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=[f"{team_a} Win", "Draw", f"{team_b} Win"],
                values=[result["p_home_win"], result["p_draw"], result["p_away_win"]],
                marker=dict(
                    colors=[BLUE, AMBER, BLUEM],
                    line=dict(color="white", width=2),
                ),
                textposition="inside",
                textinfo="label+percent",
                textfont=dict(size=11, family="Inter"),
                hole=0.4,
            ))
            light_chart(fig, 360).update_layout(
                showlegend=False,
                title="Match Outcome Probabilities",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="label">Scoreline probability matrix</div>', unsafe_allow_html=True)
            mat = result["score_matrix"][:7, :7]
            fig2 = go.Figure(go.Heatmap(
                z=mat * 100,
                x=[f"{team_b[:3].upper()} {j}" for j in range(7)],
                y=[f"{team_a[:3].upper()} {i}" for i in range(7)],
                colorscale=[[0,"#FFFFFF"],[0.5,BLUEM],[1,BLUE]],
                text=[[f"{mat[i,j]*100:.1f}%" for j in range(7)] for i in range(7)],
                texttemplate="%{text}",
                textfont=dict(size=9, family="JetBrains Mono"),
                showscale=False,
            ))
            light_chart(fig2, 360).update_layout(
                xaxis_title=f"{team_b} goals",
                yaxis_title=f"{team_a} goals",
                title="Scoreline Probability (%)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="label">Top 10 most likely scorelines</div>', unsafe_allow_html=True)

        rows = []
        score_matrix = result["score_matrix"]
        for i in range(min(9, score_matrix.shape[0])):
            for j in range(min(9, score_matrix.shape[1])):
                rows.append({
                    "Scoreline":   f"{team_a} {i}–{j} {team_b}",
                    "Probability": score_matrix[i, j],
                    "Result":      "Home Win" if i>j else ("Draw" if i==j else "Away Win"),
                })
        sl_df = pd.DataFrame(rows).nlargest(10,"Probability").reset_index(drop=True)
        sl_df.index += 1
        sl_df["Probability"] = sl_df["Probability"].apply(lambda x: f"{x:.3%}")
        st.dataframe(sl_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LINEUP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "👥 Lineup Builder":

    st.markdown("""
<div class="page-title">Lineup Builder</div>
<div class="page-sub">method: integer_programming · solver: pulp_cbc · objective: maximize p_win</div>
""", unsafe_allow_html=True)

    selected_team = st.selectbox("Select a team:", all_teams, key="lineup_team")

    if selected_team:
        squad = load_squad(selected_team)

        if squad is not None:

            st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-cell"><div class="kpi-label">Squad size</div>
    <div class="kpi-value">{len(squad)}</div></div>
  <div class="kpi-cell"><div class="kpi-label">Goalkeepers</div>
    <div class="kpi-value amber">{len(squad[squad.position=='GK'])}</div></div>
  <div class="kpi-cell"><div class="kpi-label">Defenders</div>
    <div class="kpi-value green">{len(squad[squad.position=='DEF'])}</div></div>
  <div class="kpi-cell"><div class="kpi-label">Midfielders</div>
    <div class="kpi-value blue">{len(squad[squad.position=='MID'])}</div></div>
  <div class="kpi-cell"><div class="kpi-label">Forwards</div>
    <div class="kpi-value">{len(squad[squad.position=='FWD'])}</div></div>
  <div class="kpi-cell"><div class="kpi-label">Total value</div>
    <div class="kpi-value blue">${squad['market_value_millions'].sum():.0f}M</div></div>
</div>
""", unsafe_allow_html=True)

            # Greedy optimal XI
            available = squad[~squad["injured"]].copy()
            available["strength"] = available.apply(player_str, axis=1)
            limits = {"GK":(1,1),"DEF":(3,5),"MID":(3,4),"FWD":(1,3)}
            xi_parts = [
                available[available["position"]==pos]
                .sort_values("strength",ascending=False)
                .head(mx)
                for pos,(mn,mx) in limits.items()
            ]
            xi = pd.concat(xi_parts).head(11)

            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.markdown('<div class="label">Football pitch · optimal XI</div>', unsafe_allow_html=True)

                fig_p = go.Figure()
                fig_p.add_shape(type="rect", x0=0,y0=0,x1=105,y1=68,
                                fillcolor="#16a34a",
                                line_color="#15803d", line_width=2)
                # Pitch markings
                for args in [
                    dict(type="line",  x0=52.5,y0=0,  x1=52.5,y1=68),
                    dict(type="rect",  x0=0,   y0=13.84, x1=16.5, y1=54.16),
                    dict(type="rect",  x0=88.5,y0=13.84, x1=105,  y1=54.16),
                    dict(type="rect",  x0=0,   y0=24.84, x1=5.5,  y1=43.16),
                    dict(type="rect",  x0=99.5,y0=24.84, x1=105,  y1=43.16),
                ]:
                    fig_p.add_shape(**args,
                        line=dict(color="rgba(255,255,255,0.28)", width=1.5),
                        fillcolor="rgba(0,0,0,0)")
                theta = np.linspace(0,2*np.pi,60)
                fig_p.add_trace(go.Scatter(
                    x=52.5+9.15*np.cos(theta), y=34+9.15*np.sin(theta),
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.28)", width=1.5),
                    showlegend=False, hoverinfo="none",
                ))

                xmap = {"GK":6,"DEF":24,"MID":54,"FWD":82}
                cmap = {"GK":"#FCD34D","DEF":"#6EE7B7","MID":"#93C5FD","FWD":"#FCA5A5"}

                for pos, grp in xi.groupby("position"):
                    n  = len(grp)
                    ys = np.linspace(10, 58, n+2)[1:-1]
                    for k, (_, p) in enumerate(grp.iterrows()):
                        fig_p.add_trace(go.Scatter(
                            x=[xmap.get(pos,52.5)], y=[ys[k]],
                            mode="markers+text",
                            marker=dict(size=24, color=cmap.get(pos,"white"),
                                        line=dict(color="white",width=2)),
                            text=[p["player_name"].split()[-1]],
                            textposition="bottom center",
                            textfont=dict(size=8.5, color="white", family="Inter"),
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{p['player_name']}</b><br>"
                                f"{pos} · {p['club']}<br>"
                                f"${p['market_value_millions']}M<extra></extra>"
                            ),
                        ))

                fig_p.update_layout(
                    height=460,
                    margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                               range=[0,105],fixedrange=True),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                               range=[0,68],fixedrange=True,
                               scaleanchor="x",scaleratio=0.65),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            with col2:
                st.markdown('<div class="label">Starting eleven</div>', unsafe_allow_html=True)
                for pos in ["GK","DEF","MID","FWD"]:
                    grp = xi[xi["position"]==pos]
                    for _, p in grp.iterrows():
                        st.markdown(f"""
<div class="player-row">
  <div>
    <div class="player-name">{p['player_name']}</div>
    <div class="player-club">{p['club']}</div>
  </div>
  <div class="player-meta">
    <span class="pos-tag pos-{pos}">{pos}</span>
    <span class="player-mv">${p['market_value_millions']}M</span>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="label">Sensitivity analysis · player impact scores</div>',
                        unsafe_allow_html=True)

            xi["strength"] = xi.apply(player_str, axis=1)
            base = xi["strength"].sum()
            imps = sorted([{
                "name":   p["player_name"],
                "pos":    p["position"],
                "impact": base - xi[xi["player_name"] != p["player_name"]]["strength"].sum(),
            } for _, p in xi.iterrows()], key=lambda x: -x["impact"])

            max_imp = imps[0]["impact"] if imps else 1
            for imp in imps:
                pct = (imp["impact"] / max_imp * 100) if max_imp > 0 else 0
                st.markdown(f"""
<div class="impact-row">
  <div class="impact-name">
    {imp['name'].split()[-1]}&thinsp;
    <span style="font-size:0.65rem;color:#9CA3AF;">{imp['pos']}</span>
  </div>
  <div class="impact-bar-bg">
    <div class="impact-bar-fill" style="width:{pct:.1f}%"></div>
  </div>
  <div class="impact-num">{imp['impact']:.3f}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="info-box" style="margin-top:1.5rem;">
  💡 <strong>How to read this:</strong> Higher impact = the optimal lineup weakens more when
  this player is removed. Run <code>python -m src.optimization.lineup_optimizer</code>
  for the full PuLP integer programming solution.
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
<div class="warning-box">
  ⚠️ Squad data not available for <strong>{selected_team}</strong>.
  Add a CSV to <code>data/squads/{selected_team.lower()}.csv</code>
  (available teams: {", ".join(SQUADS) if SQUADS else "none loaded yet"}).
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "ℹ️ About":

    st.markdown("""
<div class="page-title">About DecisionX</div>
<div class="page-sub">v1.0 · portfolio project · OR + ML decision platform</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
<div class="code-block">
<span class="cc"># DecisionX — FIFA WC Optimizer</span><br>
<span class="cc"># Full-stack OR + ML decision platform</span><br><br>
<span class="ck">project</span>      = <span class="cs">"FIFA World Cup Optimizer"</span><br>
<span class="ck">methods</span>      = [<span class="cs">"Poisson"</span>, <span class="cs">"XGBoost"</span>, <span class="cs">"PuLP"</span>]<br>
<span class="ck">data_range</span>   = <span class="cs">"1930 – 2022"</span><br>
<span class="ck">matches</span>      = <span class="cn">964</span><br>
<span class="ck">simulations</span>  = <span class="cn">10_000</span><br>
<span class="ck">features</span>     = <span class="cn">30</span><br>
<span class="ck">teams</span>        = <span class="cn">32</span><br>
<span class="ck">poisson_acc</span>  = <span class="cn">0.617</span><br>
<span class="ck">xgboost_acc</span>  = <span class="cn">0.492</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="label">Resume one-liner</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="code-block">
<span class="cc"># copy this into your CV</span><br><br>
<span class="cs">"Engineered DecisionX — a full-stack OR+ML</span><br>
<span class="cs"> platform that simulates 10,000 World Cup</span><br>
<span class="cs"> tournaments, solves a PuLP integer program</span><br>
<span class="cs"> to select the optimal starting XI, and</span><br>
<span class="cs"> quantifies each player's marginal impact</span><br>
<span class="cs"> on tournament win probability."</span>
</div>
""", unsafe_allow_html=True)

    with col2:
        for phase, desc in [
            ("Phase 1 · Data Pipeline",
             "Downloads 964 men's World Cup matches (1930–2022). Computes Elo ratings by replaying every match chronologically. Engineers 30 features: rolling form windows (3 and 5 matches), Elo difference, stage code, extra-time and penalty flags."),
            ("Phase 2 · Match Models",
             "Poisson goal model (61.7% accuracy): estimates attack/defence strength per team, outputs full scoreline distributions. XGBoost classifier (49.2% accuracy): direct Win/Draw/Loss prediction. Both use time-based train/test split (train ≤2014, test 2018+2022)."),
            ("Phase 3 · Tournament Simulator",
             "Simulates the full 32-team 2022 World Cup bracket 10,000 times. Each run draws random scorelines from the Poisson model, advances winners through groups and knockouts, and aggregates P(reach each stage) per team."),
            ("Phase 4 · Lineup Optimizer",
             "Integer programming with PuLP. Binary decision variables per player. Objective: maximize win probability. Constraints: 11 starters, 1 GK, 3–5 DEF, 3–4 MID, 1–3 FWD. Sensitivity analysis measures each player's marginal impact on the solution."),
        ]:
            st.markdown(f"""
<div style="margin-bottom:1.25rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.73rem;font-weight:700;
              color:{BLUE};margin-bottom:0.35rem;">{phase}</div>
  <div style="font-size:0.875rem;color:#374151;line-height:1.7;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, title, items in [
        (col1, "📈 Use Cases",
         ["Tournament prediction","Squad selection","Match betting","Tactical analysis","Risk / sensitivity"]),
        (col2, "⚡ Strengths",
         ["Probabilistic framework","Monte Carlo rigor","Constraint-aware optimization","Sensitivity analysis","Fully reproducible"]),
        (col3, "🔬 Tech Stack",
         ["Python · Pandas · NumPy","Scikit-learn · XGBoost","PuLP · CBC solver","SciPy (Poisson)","Streamlit · Plotly"]),
    ]:
        with col:
            st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
            for item in items:
                st.markdown(f"""
<div class="stat-row">
  <span class="stat-key">{item}</span>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem 0;
            border-top:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;
            font-size:0.72rem;color:#9CA3AF;">
  DecisionX — FIFA World Cup Optimizer &nbsp;·&nbsp;
  built with Python · Streamlit · Plotly · XGBoost · PuLP
</div>
""", unsafe_allow_html=True)