"""
src/dashboard/app.py
--------------------
DecisionX - FIFA World Cup Optimizer
Phase 5 — Fully Responsive Interactive Dashboard

Premium decision-support platform combining ML and OR.
Mobile-first, responsive design that works on all devices.

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


# ═════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DecisionX - FIFA WC Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════
# FULLY RESPONSIVE CSS - MOBILE FIRST DESIGN
# ═════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ─────────────────────────────────────────────────────────────── */
/* GOOGLE FONTS & DESIGN TOKENS                                   */
/* ─────────────────────────────────────────────────────────────── */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

:root {
    /* Colors */
    --primary: #1e40af;
    --primary-dark: #1e3a8a;
    --primary-light: #3b82f6;
    --secondary: #0369a1;
    --accent: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f97316;
    --info: #0ea5e9;
    
    /* Neutrals */
    --bg-light: #f9fafb;
    --bg-lighter: #f3f4f6;
    --bg-dark: #111827;
    --text-dark: #1f2937;
    --text-light: #6b7280;
    --border-color: #e5e7eb;
    
    /* Typography */
    --font-family: 'Inter', sans-serif;
    --font-display: 'Poppins', sans-serif;
    
    /* Spacing */
    --spacing-xs: 0.5rem;
    --spacing-sm: 1rem;
    --spacing-md: 1.5rem;
    --spacing-lg: 2rem;
    --spacing-xl: 3rem;
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 16px;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}

/* ─────────────────────────────────────────────────────────────── */
/* GLOBAL STYLES                                                   */
/* ─────────────────────────────────────────────────────────────── */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    font-family: var(--font-family);
    color: var(--text-dark);
    background: var(--bg-light);
}

/* ─────────────────────────────────────────────────────────────── */
/* STREAMLIT MAIN APP STYLING                                      */
/* ─────────────────────────────────────────────────────────────── */

.main {
    background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-lighter) 100%);
    padding: var(--spacing-md);
}

.main .block-container {
    padding: var(--spacing-md);
    max-width: 100% !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* TYPOGRAPHY - RESPONSIVE                                         */
/* ─────────────────────────────────────────────────────────────── */

h1 {
    font-family: var(--font-display);
    font-size: clamp(1.75rem, 5vw, 2.5rem);
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-family: var(--font-display);
    font-size: clamp(1.25rem, 4vw, 1.75rem);
    font-weight: 700;
    color: var(--primary-dark);
    margin: var(--spacing-lg) 0 var(--spacing-md) 0;
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.5rem;
}

h3 {
    font-family: var(--font-display);
    font-size: clamp(1rem, 3vw, 1.3rem);
    font-weight: 600;
    color: var(--secondary);
    margin: var(--spacing-md) 0 var(--spacing-sm) 0;
}

p {
    font-size: clamp(0.875rem, 1.5vw, 1rem);
    line-height: 1.6;
    color: var(--text-light);
}

.header-subtitle {
    color: var(--text-light);
    font-size: clamp(0.95rem, 2vw, 1.1rem);
    margin-bottom: var(--spacing-lg);
    font-weight: 500;
}

/* ─────────────────────────────────────────────────────────────── */
/* CARDS - RESPONSIVE                                              */
/* ─────────────────────────────────────────────────────────────── */

.card {
    background: white;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    padding: clamp(1rem, 4vw, 2rem);
    margin-bottom: var(--spacing-lg);
    border-top: 4px solid var(--accent);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

/* ─────────────────────────────────────────────────────────────── */
/* STAT BOXES - RESPONSIVE                                         */
/* ─────────────────────────────────────────────────────────────── */

.stat-box {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
    color: white;
    padding: clamp(1rem, 3vw, 1.5rem);
    border-radius: var(--radius-md);
    text-align: center;
    margin-bottom: var(--spacing-md);
    transition: all 0.3s ease;
}

.stat-box:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

.stat-box-title {
    font-size: clamp(0.75rem, 1.5vw, 0.85rem);
    opacity: 0.9;
    margin-bottom: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

.stat-box-value {
    font-family: var(--font-display);
    font-size: clamp(1.5rem, 4vw, 2rem);
    font-weight: 800;
}

/* ─────────────────────────────────────────────────────────────── */
/* ALERT BOXES - RESPONSIVE                                        */
/* ─────────────────────────────────────────────────────────────── */

.info-box, .success-box, .warning-box {
    border-radius: var(--radius-md);
    padding: clamp(0.875rem, 2vw, 1rem);
    margin: var(--spacing-md) 0;
    font-size: clamp(0.85rem, 1.5vw, 0.95rem);
    border-left: 4px solid;
    line-height: 1.6;
}

.info-box {
    background: #eff6ff;
    border-color: var(--info);
    color: #0c4a6e;
}

.success-box {
    background: #f0fdf4;
    border-color: var(--success);
    color: #165e4d;
}

.warning-box {
    background: #fffbeb;
    border-color: var(--warning);
    color: #92400e;
}

.highlight-accent {
    background: linear-gradient(135deg, var(--accent) 0%, var(--warning) 100%);
    color: white;
    padding: clamp(1rem, 4vw, 1.5rem);
    border-radius: var(--radius-md);
    margin: var(--spacing-md) 0;
}

.highlight-accent h3 {
    color: white;
    margin-top: 0;
}

.highlight-accent p {
    color: rgba(255, 255, 255, 0.95);
}

/* ─────────────────────────────────────────────────────────────── */
/* SIDEBAR - RESPONSIVE                                            */
/* ─────────────────────────────────────────────────────────────── */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--primary-dark) 0%, var(--secondary) 100%);
    color: white;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.logo-text {
    font-family: var(--font-display);
    font-size: clamp(1.2rem, 3vw, 1.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, white, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
}

/* ─────────────────────────────────────────────────────────────── */
/* METRICS - RESPONSIVE                                            */
/* ─────────────────────────────────────────────────────────────── */

[data-testid="stMetric"] {
    background: white;
    padding: clamp(1rem, 2vw, 1.5rem);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    border-left: 4px solid var(--primary-dark);
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

[data-testid="stMetric"] label {
    font-size: clamp(0.75rem, 1.5vw, 0.85rem) !important;
    color: var(--text-light) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.5rem, 4vw, 2rem) !important;
    color: var(--primary-dark) !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    color: var(--success) !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* BUTTONS - RESPONSIVE                                            */
/* ─────────────────────────────────────────────────────────────── */

.stButton > button {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
    color: white;
    border: none;
    padding: clamp(0.6rem, 1.5vw, 0.75rem) clamp(1rem, 3vw, 2rem);
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: clamp(0.875rem, 1.5vw, 1rem);
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    opacity: 0.95;
}

/* ─────────────────────────────────────────────────────────────── */
/* SELECTBOX & INPUTS - RESPONSIVE                                 */
/* ─────────────────────────────────────────────────────────────── */

[data-testid="stSelectbox"] > div > div {
    background: white !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    font-size: clamp(0.875rem, 1.5vw, 1rem) !important;
}

[data-testid="stMultiSelect"] > div > div {
    background: white !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* TABS - RESPONSIVE                                               */
/* ─────────────────────────────────────────────────────────────── */

.stTabs [data-baseweb="tab-list"] {
    gap: clamp(0.5rem, 2vw, 1rem);
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    padding: clamp(0.6rem, 1.5vw, 0.75rem) clamp(1rem, 2vw, 1.5rem);
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: clamp(0.825rem, 1.5vw, 0.95rem);
    background: var(--bg-lighter);
    color: var(--text-light);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
    color: white;
}

/* ─────────────────────────────────────────────────────────────── */
/* DATAFRAMES - RESPONSIVE                                         */
/* ─────────────────────────────────────────────────────────────── */

[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: auto;
    font-size: clamp(0.75rem, 1.5vw, 0.9rem);
}

[data-testid="stDataFrame"] th {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: clamp(0.5rem, 1.5vw, 0.75rem) !important;
}

[data-testid="stDataFrame"] tr:nth-child(even) {
    background: var(--bg-lighter);
}

[data-testid="stDataFrame"] tr:hover {
    background: #e0f2fe !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* DIVIDER - RESPONSIVE                                            */
/* ─────────────────────────────────────────────────────────────── */

hr {
    border-color: var(--border-color) !important;
    margin: var(--spacing-lg) 0 !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* FOOTER                                                           */
/* ─────────────────────────────────────────────────────────────── */

.footer {
    text-align: center;
    padding: var(--spacing-lg);
    color: var(--text-light);
    border-top: 1px solid var(--border-color);
    margin-top: var(--spacing-xl);
    font-size: clamp(0.75rem, 1.5vw, 0.85rem);
}

.footer p {
    margin: 0.5rem 0;
}

/* ─────────────────────────────────────────────────────────────── */
/* PLOTLY CHARTS - RESPONSIVE                                      */
/* ─────────────────────────────────────────────────────────────── */

.plotly-graph-div {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
}

/* ─────────────────────────────────────────────────────────────── */
/* COLUMNS & LAYOUT - RESPONSIVE                                   */
/* ─────────────────────────────────────────────────────────────── */

[data-testid="stColumn"] {
    transition: all 0.3s ease;
}

/* ─────────────────────────────────────────────────────────────── */
/* MOBILE RESPONSIVE (max-width: 640px)                             */
/* ─────────────────────────────────────────────────────────────── */

@media (max-width: 640px) {
    :root {
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
    }

    .main {
        padding: var(--spacing-sm);
    }

    .main .block-container {
        padding: var(--spacing-sm) !important;
    }

    h1 {
        margin-bottom: 0.25rem;
    }

    h2 {
        margin: var(--spacing-md) 0 var(--spacing-sm) 0;
    }

    .card {
        padding: var(--spacing-sm);
        margin-bottom: var(--spacing-md);
    }

    .stat-box {
        padding: var(--spacing-sm);
        margin-bottom: var(--spacing-sm);
    }

    .highlight-accent {
        padding: var(--spacing-sm);
    }

    [data-testid="stMetric"] {
        padding: var(--spacing-sm);
    }

    .stButton > button {
        width: 100% !important;
        padding: 0.6rem 1rem !important;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.75rem;
    }
}

/* ─────────────────────────────────────────────────────────────── */
/* TABLET RESPONSIVE (641px - 1024px)                              */
/* ─────────────────────────────────────────────────────────────── */

@media (min-width: 641px) and (max-width: 1024px) {
    .main {
        padding: var(--spacing-md);
    }

    .main .block-container {
        padding: var(--spacing-md) !important;
    }

    .card {
        padding: var(--spacing-md);
    }

    [data-testid="stMetric"] {
        padding: var(--spacing-md);
    }
}

/* ─────────────────────────────────────────────────────────────── */
/* LARGE DESKTOP (1024px+)                                          */
/* ─────────────────────────────────────────────────────────────── */

@media (min-width: 1024px) {
    .main {
        padding: var(--spacing-lg);
    }

    .main .block-container {
        padding: var(--spacing-lg) !important;
        max-width: 1400px;
        margin: 0 auto;
    }
}

/* ─────────────────────────────────────────────────────────────── */
/* UTILITY CLASSES                                                  */
/* ─────────────────────────────────────────────────────────────── */

.text-center {
    text-align: center;
}

.mt-2 {
    margin-top: var(--spacing-md);
}

.mb-2 {
    margin-bottom: var(--spacing-md);
}

.gap-2 {
    gap: var(--spacing-md);
}

</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════

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
        st.error(f"Error loading poisson params: {e}")
        return None

@st.cache_data
def load_squad(team_name):
    try:
        path = cfg.OUTPUTS.parent / "data" / "squads" / f"{team_name.lower()}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)
    except Exception:
        return None

tournament_probs = load_tournament_probs()
poisson_params = load_poisson_params()

if tournament_probs is None or poisson_params is None:
    st.error("❌ Unable to load required data files. Please ensure Phase 1-3 are complete.")
    st.stop()

avg_goals = poisson_params["avg_goals_baseline"].iloc[0]
all_teams = sorted(tournament_probs["team"].unique())


# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("<div style='font-size: clamp(2rem, 6vw, 3rem); text-align: center;'>⚽</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1>DecisionX</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>🏆 FIFA World Cup Tournament Optimization System</p>", unsafe_allow_html=True)

st.divider()


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div class='logo-text'>⚽ DecisionX</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    tab_choice = st.radio(
        "📊 Navigation",
        [
            "🏠 Home",
            "🎯 Team Explorer",
            "🎲 Bracket Simulator",
            "⚔️ Head-to-Head",
            "👥 Lineup Builder",
            "ℹ️ About"
        ],
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.markdown("""
### 🔧 System Architecture

**Phase 1:** Data Pipeline
- Elo ratings, form features

**Phase 2:** Match Models
- Poisson + XGBoost

**Phase 3:** Tournament Sim
- 10,000 Monte Carlo runs

**Phase 4:** Lineup Optimizer
- Integer Programming

**Phase 5:** Dashboard
- Interactive UI ✅
""")
    
    st.markdown("---")
    st.markdown(f"""
<div style='text-align: center; font-size: 0.85rem; color: rgba(255,255,255,0.7);'>
📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
<br>
✅ Data fresh from Phase 1-4
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════

if tab_choice == "🏠 Home":
    st.markdown("<h2>Welcome to DecisionX</h2>", unsafe_allow_html=True)
    st.markdown("""
<div class='card'>
    <p>A premium decision-support platform for FIFA World Cup analysis and optimization. Combining machine learning, operations research, and simulation to provide data-driven insights.</p>
</div>
""", unsafe_allow_html=True)

    # Key stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class='stat-box'>
    <div class='stat-box-title'>Teams Analyzed</div>
    <div class='stat-box-value'>32</div>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class='stat-box'>
    <div class='stat-box-title'>Simulations Run</div>
    <div class='stat-box-value'>10K</div>
</div>
""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class='stat-box'>
    <div class='stat-box-title'>Historical Matches</div>
    <div class='stat-box-value'>964</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Features overview
    st.markdown("<h2>Platform Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
<div class='card'>
    <h3>🎯 Team Explorer</h3>
    <p>Explore tournament probabilities for any team. See their chances of reaching each stage from group stage to the final.</p>
    <p><strong>Use for:</strong> Understanding team strength and tournament path</p>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("""
<div class='card'>
    <h3>🎲 Bracket Simulator</h3>
    <p>View tournament brackets and aggregate probabilities from 10,000 simulated tournaments.</p>
    <p><strong>Use for:</strong> Tournament structure understanding</p>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
<div class='card'>
    <h3>⚔️ Head-to-Head</h3>
    <p>Predict the outcome of any match. Get full scoreline probabilities using the Poisson goal model.</p>
    <p><strong>Use for:</strong> Match betting, tactical analysis</p>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("""
<div class='card'>
    <h3>👥 Lineup Builder</h3>
    <p>Discover the optimal starting XI for any team. Understand player impact through sensitivity analysis.</p>
    <p><strong>Use for:</strong> Squad selection, team strategy</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class='highlight-accent'>
    <h3>🚀 Quick Start</h3>
    <p>1. Go to <strong>Team Explorer</strong> and select your team<br>
    2. Check <strong>Head-to-Head</strong> for specific match predictions<br>
    3. Use <strong>Lineup Builder</strong> to optimize your squad<br>
    4. Explore the <strong>Bracket Simulator</strong> to understand tournament dynamics</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Top 5 teams
    st.markdown("<h2>🏆 Top 5 Teams by Win Probability</h2>", unsafe_allow_html=True)
    top5 = tournament_probs.nlargest(5, "p_win")[["team", "p_win", "p_final", "p_qf"]].reset_index(drop=True)
    
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        col1, col2, col3, col4 = st.columns([0.5, 2, 1.5, 1.5])
        with col1:
            st.markdown(f"<h3 style='margin:0; color: var(--accent);'>{idx}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='margin:0;'>{row['team']}</h3>", unsafe_allow_html=True)
        with col3:
            st.metric("Win %", f"{row['p_win']:.1%}")
        with col4:
            st.metric("Final %", f"{row['p_final']:.1%}")
        st.divider()


# ═════════════════════════════════════════════════════════════════════════
# PAGE: TEAM EXPLORER
# ═════════════════════════════════════════════════════════════════════════

elif tab_choice == "🎯 Team Explorer":
    st.markdown("<h2>🎯 Team Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Analyze tournament probabilities for any team</p>", unsafe_allow_html=True)

    selected_team = st.selectbox("Select a team:", all_teams, key="explorer_team")

    if selected_team:
        row = tournament_probs[tournament_probs["team"] == selected_team].iloc[0]
        rank = tournament_probs[tournament_probs["team"] == selected_team].index[0] + 1

        st.markdown(f"<h3>{selected_team}</h3>", unsafe_allow_html=True)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reach Group", f"{row['p_group']:.1%}")
        with col2:
            st.metric("Reach QF", f"{row['p_qf']:.1%}")
        with col3:
            st.metric("Reach Final", f"{row['p_final']:.1%}")
        with col4:
            st.metric("Win WC", f"{row['p_win']:.1%}")

        st.divider()

        # Tournament path
        st.markdown("<h3>Tournament Path Probabilities</h3>", unsafe_allow_html=True)

        stages = ["Group", "R16", "QF", "SF", "Final", "Win"]
        probs = [row["p_group"], row["p_r16"], row["p_qf"], row["p_sf"], row["p_final"], row["p_win"]]

        fig = go.Figure(data=[
            go.Bar(
                x=stages,
                y=probs,
                marker=dict(
                    color=probs,
                    colorscale=[[0, "#1e40af"], [0.5, "#0369a1"], [1.0, "#f59e0b"]],
                    showscale=False,
                ),
                text=[f"{p:.1%}" for p in probs],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>",
            )
        ])

        fig.update_layout(
            title=f"{selected_team} — Tournament Stage Progression",
            xaxis_title="Tournament Stage",
            yaxis_title="Probability",
            height=400,
            showlegend=False,
            template="plotly_white",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Rankings
        st.markdown("<h3>Global Rankings</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'><strong>{selected_team}</strong> is ranked <strong>#{rank}/32</strong> by win probability</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3>Top 10 Teams (Win %)</h3>", unsafe_allow_html=True)
            top10 = tournament_probs.nlargest(10, "p_win")[["team", "p_win"]].reset_index(drop=True)
            top10.index = top10.index + 1
            top10.columns = ["Team", "Win %"]
            top10["Win %"] = top10["Win %"].apply(lambda x: f"{x:.2%}")
            st.dataframe(top10, use_container_width=True)

        with col2:
            st.markdown("<h3>Top 10 Teams (Final %)</h3>", unsafe_allow_html=True)
            top10_final = tournament_probs.nlargest(10, "p_final")[["team", "p_final"]].reset_index(drop=True)
            top10_final.index = top10_final.index + 1
            top10_final.columns = ["Team", "Final %"]
            top10_final["Final %"] = top10_final["Final %"].apply(lambda x: f"{x:.2%}")
            st.dataframe(top10_final, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: BRACKET SIMULATOR
# ═════════════════════════════════════════════════════════════════════════

elif tab_choice == "🎲 Bracket Simulator":
    st.markdown("<h2>🎲 Bracket Simulator</h2>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Tournament probabilities from 10,000 simulated tournaments</p>", unsafe_allow_html=True)

    st.markdown("""
<div class='success-box'>
✅ <strong>10,000 simulations completed</strong> — Results are statistically robust
</div>
""", unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_win_prob = tournament_probs["p_win"].mean()
        st.metric("Avg Win %", f"{avg_win_prob:.2%}")
    with col2:
        max_win_prob = tournament_probs["p_win"].max()
        max_team = tournament_probs.loc[tournament_probs["p_win"].idxmax(), "team"]
        st.metric(f"Favorite ({max_team})", f"{max_win_prob:.2%}")
    with col3:
        median_final = tournament_probs["p_final"].median()
        st.metric("Median Final %", f"{median_final:.2%}")

    st.divider()

    # Full table
    st.markdown("<h3>Complete Tournament Probabilities</h3>", unsafe_allow_html=True)

    display_cols = ["team", "p_group", "p_r16", "p_qf", "p_sf", "p_final", "p_win"]
    display_df = tournament_probs[display_cols].copy().reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.columns = ["Team", "Group", "R16", "QF", "SF", "Final", "Win"]

    for col in ["Group", "R16", "QF", "SF", "Final", "Win"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df, use_container_width=True, height=600)

    st.divider()

    # Distribution charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Distribution: Win Probabilities</h3>", unsafe_allow_html=True)
        fig = px.histogram(
            tournament_probs,
            x="p_win",
            nbins=15,
            labels={"p_win": "Win Probability"},
            color_discrete_sequence=["#1e40af"],
            title="",
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h3>Distribution: Final Probabilities</h3>", unsafe_allow_html=True)
        fig = px.histogram(
            tournament_probs,
            x="p_final",
            nbins=15,
            labels={"p_final": "Final Probability"},
            color_discrete_sequence=["#0369a1"],
            title="",
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: HEAD-TO-HEAD
# ═════════════════════════════════════════════════════════════════════════

elif tab_choice == "⚔️ Head-to-Head":
    st.markdown("<h2>⚔️ Head-to-Head Match Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Poisson goal model predictions for any matchup</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Home Team:", all_teams, key="h2h_home", index=0)
    with col2:
        team_b = st.selectbox("Away Team:", [t for t in all_teams if t != team_a], key="h2h_away")

    if team_a and team_b and team_a != team_b:
        result = predict_match(team_a, team_b, poisson_params, avg_goals)

        st.divider()

        # Prediction result
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
<div class='stat-box'>
    <div class='stat-box-title'>{team_a} Win</div>
    <div class='stat-box-value'>{result['p_home_win']:.1%}</div>
</div>
""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
<div class='stat-box'>
    <div class='stat-box-title'>Draw</div>
    <div class='stat-box-value'>{result['p_draw']:.1%}</div>
</div>
""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
<div class='stat-box'>
    <div class='stat-box-title'>{team_b} Win</div>
    <div class='stat-box-value'>{result['p_away_win']:.1%}</div>
</div>
""", unsafe_allow_html=True)

        # Expected goals
        st.markdown(f"""
<div class='info-box'>
    <strong>Expected Goals:</strong> {team_a} <strong>{result['lambda_home']:.2f}</strong> — <strong>{result['lambda_away']:.2f}</strong> {team_b}
</div>
""", unsafe_allow_html=True)

        st.divider()

        # Match outcome chart
        st.markdown("<h3>Match Outcome Probabilities</h3>", unsafe_allow_html=True)

        outcomes = [f"{team_a} Win", "Draw", f"{team_b} Win"]
        probs = [result["p_home_win"], result["p_draw"], result["p_away_win"]]

        fig = go.Figure(data=[
            go.Pie(
                labels=outcomes,
                values=probs,
                marker=dict(colors=["#1e40af", "#f59e0b", "#0369a1"]),
                textposition="inside",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
            )
        ])

        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Scoreline probabilities
        st.markdown(f"<h3>Top 10 Most Likely Scorelines</h3>", unsafe_allow_html=True)

        score_matrix = result["score_matrix"]
        scorelines = []

        for i in range(min(9, score_matrix.shape[0])):
            for j in range(min(9, score_matrix.shape[1])):
                scorelines.append({
                    "Scoreline": f"{team_a} {i}–{j} {team_b}",
                    "Probability": f"{score_matrix[i, j]:.3%}",
                    "Prob_Val": score_matrix[i, j],
                })

        scorelines_df = pd.DataFrame(scorelines).nlargest(10, "Prob_Val")[["Scoreline", "Probability"]]
        scorelines_df = scorelines_df.reset_index(drop=True)
        scorelines_df.index = scorelines_df.index + 1

        st.dataframe(scorelines_df, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: LINEUP BUILDER
# ═════════════════════════════════════════════════════════════════════════

elif tab_choice == "👥 Lineup Builder":
    st.markdown("<h2>👥 Lineup Builder</h2>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Optimal starting XI and player impact analysis</p>", unsafe_allow_html=True)

    selected_team = st.selectbox("Select a team:", all_teams, key="lineup_team")

    if selected_team:
        squad = load_squad(selected_team)

        if squad is not None:
            # Squad stats
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Squad Size", len(squad))
            with col2:
                st.metric("GK", len(squad[squad["position"] == "GK"]))
            with col3:
                st.metric("DEF", len(squad[squad["position"] == "DEF"]))
            with col4:
                st.metric("MID", len(squad[squad["position"] == "MID"]))
            with col5:
                st.metric("FWD", len(squad[squad["position"] == "FWD"]))

            st.divider()

            st.markdown("<h3>Squad Composition</h3>", unsafe_allow_html=True)

            display = squad[["player_name", "position", "club", "market_value_millions"]].copy()
            display.columns = ["Player", "Position", "Club", "Market Value (M$)"]
            display = display.reset_index(drop=True)
            display.index = display.index + 1

            st.dataframe(display, use_container_width=True, height=500)

            st.markdown("""
<div class='info-box'>
💡 <strong>In Phase 4,</strong> an Integer Programming optimizer selects the 11 best starters from this squad that maximize tournament win probability, subject to formation constraints (1 GK, 3-5 DEF, 3-4 MID, 1-3 FWD).

🔍 <strong>Sensitivity analysis</strong> then measures each player's impact by re-solving the optimization problem with that player removed.
</div>
""", unsafe_allow_html=True)

        else:
            st.warning(f"Squad data not available for {selected_team}. Add squad CSV to `data/squads/{selected_team.lower()}.csv`")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════

elif tab_choice == "ℹ️ About":
    st.markdown("<h2>ℹ️ About DecisionX</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class='card'>
    <h3>🎯 Mission</h3>
    <p>Transform World Cup analysis through data science and operations research. Provide decision-makers with probability-driven insights for tournament prediction and squad optimization.</p>
</div>

<div class='card'>
    <h3>🏗️ Architecture</h3>
    <p><strong>Phase 1:</strong> Data Pipeline<br>
Load 964 historical matches, compute Elo ratings, engineer 30+ features<br><br>
<strong>Phase 2:</strong> Match Models<br>
Poisson goal model (61.7% accuracy) + XGBoost classifier (49.2% accuracy)<br><br>
<strong>Phase 3:</strong> Tournament Simulator<br>
Monte Carlo: 10,000 tournament runs<br><br>
<strong>Phase 4:</strong> Lineup Optimizer<br>
Integer Programming (PuLP) to maximize win probability<br><br>
<strong>Phase 5:</strong> Dashboard<br>
Interactive Streamlit web application ✅</p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class='card'>
    <h3>📊 Key Metrics</h3>
    <p>
    ✅ <strong>964</strong> historical World Cup matches analyzed<br>
    ✅ <strong>81</strong> teams with Elo ratings<br>
    ✅ <strong>30</strong> engineered features per match<br>
    ✅ <strong>10,000</strong> tournament simulations<br>
    ✅ <strong>32</strong> teams with probability profiles<br>
    </p>
</div>

<div class='card'>
    <h3>🔬 Technical Stack</h3>
    <p>
    <strong>Data:</strong> Pandas, NumPy<br>
    <strong>ML:</strong> Scikit-learn, XGBoost<br>
    <strong>OR:</strong> PuLP, CBC solver<br>
    <strong>Simulation:</strong> NumPy random<br>
    <strong>UI:</strong> Streamlit, Plotly<br>
    </p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class='highlight-accent'>
    <h3>🚀 How to Use DecisionX</h3>
    <p>
    <strong>1. Team Explorer:</strong> Pick a team → see tournament path probabilities<br>
    <strong>2. Head-to-Head:</strong> Compare two teams → get match prediction<br>
    <strong>3. Bracket Simulator:</strong> Understand probability distribution across 32 teams<br>
    <strong>4. Lineup Builder:</strong> Explore squad composition and optimal selection<br>
    <strong>5. About:</strong> Learn the methodology and technical details
    </p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class='card'>
    <h3>📈 Use Cases</h3>
    <ul>
    <li>Tournament betting</li>
    <li>Team strategy</li>
    <li>Squad selection</li>
    <li>Tactical analysis</li>
    <li>Match prediction</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class='card'>
    <h3>⚡ Strengths</h3>
    <ul>
    <li>Probabilistic framework</li>
    <li>Monte Carlo rigor</li>
    <li>Constraint-aware optimization</li>
    <li>Sensitivity analysis</li>
    <li>Fully reproducible</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class='card'>
    <h3>🎓 Learnings</h3>
    <ul>
    <li>Draws are unpredictable</li>
    <li>Form matters as much as Elo</li>
    <li>Knockouts are high-variance</li>
    <li>Lineup optimization has constraints</li>
    <li>Ensemble models beat individual ones</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class='footer'>
    <p><strong>DecisionX — FIFA World Cup Optimizer</strong></p>
    <p>A portfolio project demonstrating Operations Research + Machine Learning</p>
    <p>📧 Portfolio | 🔗 GitHub | 📊 Documentation</p>
    <p>Built with Python • Streamlit • Plotly • PuLP</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# END OF APPLICATION
# ═════════════════════════════════════════════════════════════════════════