"""
EcomIQ — Sales Intelligence Platform
=====================================
A premium dark-mode Streamlit application for e-commerce analytics
and customer churn prediction using the Olist Brazilian E-Commerce dataset.

Author: Soumendra Brahmapada
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable regardless of launch directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_datasets, merge_master_dataframe, get_data_summary
from src.feature_engineering import build_feature_matrix
from src.model import (
    load_model, train_and_compare_all, evaluate_model,
    generate_shap_values, save_model,
)
from src.visualizations import (
    create_monthly_revenue_trend,
    create_top_categories_chart,
    create_order_status_distribution,
    create_state_distribution,
    create_review_distribution,
    create_payment_breakdown,
    create_delivery_performance,
    create_churn_gauge,
    create_roc_curves,
    create_confusion_matrix_chart,
    create_feature_importance,
)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the first Streamlit command
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EcomIQ — Sales Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Full Custom CSS Injection
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Google Font ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Fade-in Animation ─────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Global Reset ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #a1a1aa;
}

/* ── Main background ───────────────────────────────────────────────── */
.stApp, .main, .block-container {
    background-color: #000000 !important;
}

.block-container {
    padding: 2rem 3rem 2rem 3rem !important;
    max-width: 1400px !important;
    animation: fadeIn 0.5s ease-out;
}

/* ── Remove Streamlit defaults ─────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}

/* ── Sidebar ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #222222 !important;
    padding-top: 1rem;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1rem 1.5rem !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
section[data-testid="stSidebar"] label {
    color: #a1a1aa !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar Radio (Navigation Pills) ──────────────────────────────── */
section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 4px;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #111111 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    margin: 2px 0 !important;
    border-left: 3px solid transparent !important;
    transition: all 0.2s ease !important;
    color: #a1a1aa !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    border-left-color: #0070f3 !important;
    background-color: #222222 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
    border-left-color: #0070f3 !important;
    background-color: #002550 !important;
    color: #ededed !important;
}

/* ── Typography ────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #ededed !important;
}

h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.1rem !important; }

p, span, div, li {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide default st.metric styling ────────────────────────────────── */
[data-testid="stMetric"] {
    display: none !important;
}

/* ── Custom KPI Cards ──────────────────────────────────────────────── */
.kpi-card {
    background: #111111;
    border: 1px solid #222222;
    border-left: 3px solid #0070f3;
    border-radius: 12px;
    padding: 20px 24px;
    transition: all 0.2s ease;
    animation: fadeIn 0.6s ease-out;
}

.kpi-card:hover {
    border-color: #0070f3;
    box-shadow: 0 4px 20px rgba(0, 112, 243, 0.08);
    transform: translateY(-2px);
}

.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0070f3;
    line-height: 1.1;
    font-family: 'Inter', sans-serif;
}

.kpi-subtitle {
    font-size: 0.75rem;
    color: #888888;
    margin-top: 4px;
    font-family: 'Inter', sans-serif;
}

/* ── Section Headers ───────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1rem 0;
    animation: fadeIn 0.5s ease-out;
}

.section-header .accent-line {
    width: 3px;
    height: 18px;
    background: #0070f3;
    border-radius: 2px;
}

.section-header .section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888888;
    font-family: 'Inter', sans-serif;
}

/* ── Chart containers ──────────────────────────────────────────────── */
.chart-container {
    background: #0a0a0a;
    border: 1px solid #222222;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    transition: all 0.2s ease;
}

.chart-container:hover {
    border-color: #374151;
}

/* ── Risk Badges ───────────────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.05em;
}

.risk-low {
    background: rgba(0, 112, 243, 0.15);
    color: #0070f3;
    border: 1px solid rgba(0, 112, 243, 0.3);
}

.risk-medium {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.risk-high {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ── Styled Tables ─────────────────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    margin: 1rem 0;
}

.styled-table thead th {
    background: #0a0a0a;
    color: #ededed;
    font-weight: 600;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid #0070f3;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.styled-table tbody td {
    padding: 10px 16px;
    color: #a1a1aa;
    border-bottom: 1px solid #111111;
}

.styled-table tbody tr:nth-child(odd) {
    background: #111111;
}

.styled-table tbody tr:nth-child(even) {
    background: #0a0a0a;
}

.styled-table tbody tr:hover {
    background: #222222;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton > button {
    background: #0070f3 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #00B89A !important;
    box-shadow: 0 4px 12px rgba(0, 112, 243, 0.25) !important;
}

/* ── File uploader ─────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #111111 !important;
    border: 1px dashed #222222 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ── Selectbox / Input ─────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: #111111 !important;
    color: #ededed !important;
    border: 1px solid #222222 !important;
    border-radius: 8px !important;
}

/* ── Spinner ───────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #0070f3 !important;
}

/* ── Divider ───────────────────────────────────────────────────────── */
hr {
    border-color: #111111 !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #000000;
}
::-webkit-scrollbar-thumb {
    background: #222222;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #888888;
}

/* ── Author Card ───────────────────────────────────────────────────── */
.author-card {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 1.5rem;
}

.author-name {
    color: #ededed;
    font-weight: 600;
    font-size: 0.85rem;
    font-family: 'Inter', sans-serif;
}

.author-role {
    color: #888888;
    font-size: 0.7rem;
    font-family: 'Inter', sans-serif;
    margin-top: 2px;
}

.author-links {
    margin-top: 8px;
    display: flex;
    gap: 10px;
}

.author-links a {
    color: #0070f3 !important;
    font-size: 0.7rem;
    text-decoration: none;
    font-family: 'Inter', sans-serif;
    transition: color 0.2s ease;
}

.author-links a:hover {
    color: #00B89A !important;
}

/* ── Expander ──────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #111111 !important;
    color: #ededed !important;
    border-radius: 8px !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: #111111 !important;
    color: #a1a1aa !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    border: 1px solid #222222 !important;
}

.stTabs [aria-selected="true"] {
    background: #222222 !important;
    color: #0070f3 !important;
    border-color: #0070f3 !important;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════
# INJECT CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_section_header(title: str) -> None:
    """Render a styled section header with teal accent line."""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="accent-line"></div>'
        f'<span class="section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str = "") -> str:
    """Return HTML for a single KPI metric card."""
    sub_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def render_risk_badge(probability: float) -> str:
    """Return an HTML risk badge based on churn probability."""
    if probability < 0.4:
        return '<span class="risk-badge risk-low">● Low Risk</span>'
    elif probability < 0.7:
        return '<span class="risk-badge risk-medium">● Medium Risk</span>'
    else:
        return '<span class="risk-badge risk-high">● High Risk</span>'


def render_model_comparison_table(results: dict) -> str:
    """Render a styled HTML table for model comparison results."""
    rows = ""
    for name, metrics in results.items():
        rows += (
            f"<tr>"
            f"<td style='color: #ededed; font-weight: 500;'>{name}</td>"
            f"<td>{metrics.get('accuracy', 0):.4f}</td>"
            f"<td>{metrics.get('precision', 0):.4f}</td>"
            f"<td>{metrics.get('recall', 0):.4f}</td>"
            f"<td>{metrics.get('f1', 0):.4f}</td>"
            f"<td style='color: #0070f3; font-weight: 600;'>"
            f"{metrics.get('roc_auc', 0):.4f}</td>"
            f"</tr>"
        )
    return (
        '<table class="styled-table">'
        "<thead><tr>"
        "<th>Model</th><th>Accuracy</th><th>Precision</th>"
        "<th>Recall</th><th>F1-Score</th><th>ROC-AUC</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load and merge all Olist datasets. Cached for performance."""
    data_dir = PROJECT_ROOT / "data" / "raw"
    datasets = load_all_datasets(data_dir)
    master_df = merge_master_dataframe(datasets)
    return master_df


@st.cache_data(show_spinner=False)
def load_feature_data():
    """Build the customer-level feature matrix. Cached for performance."""
    df = load_data()
    features = build_feature_matrix(df)
    return features


@st.cache_resource(show_spinner=False)
def load_trained_models():
    """Load or train ML models. Returns models dict and results dict."""
    from sklearn.model_selection import train_test_split

    features = load_feature_data()
    feature_cols = [
        "frequency", "monetary",
        "avg_review_score", "avg_delivery_days", "late_delivery_rate",
        "unique_categories", "preferred_payment",
    ]

    # Ensure all feature columns exist
    available_cols = [c for c in feature_cols if c in features.columns]
    X = features[available_cols]
    y = features["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_path = PROJECT_ROOT / "models" / "churn_model.pkl"

    # Try to load pre-trained models
    models_cache = PROJECT_ROOT / "models" / "all_models.pkl"
    results_cache = PROJECT_ROOT / "models" / "all_results.pkl"

    try:
        import joblib
        if models_cache.exists() and results_cache.exists():
            models = joblib.load(models_cache)
            results = joblib.load(results_cache)
            return models, results, X_train, X_test, y_train, y_test, available_cols
    except Exception:
        pass

    # Train fresh
    models, results = train_and_compare_all(X_train, X_test, y_train, y_test)

    # Save models
    try:
        import joblib
        os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
        save_model(models["XGBoost"], model_path)
        joblib.dump(models, models_cache)
        joblib.dump(results, results_cache)
    except Exception:
        pass

    return models, results, X_train, X_test, y_train, y_test, available_cols


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo
    st.markdown(
        '<div style="padding: 0.5rem 0 0.2rem 0;">'
        '<span style="color: #0070f3; font-size: 1.4rem; font-weight: 700; '
        'font-family: Inter, sans-serif;">◈ EcomIQ</span>'
        '</div>'
        '<div style="color: #888888; font-size: 0.75rem; font-family: Inter, sans-serif; '
        'margin-bottom: 1.5rem; letter-spacing: 0.03em;">'
        'Sales Intelligence Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="color: #888888; font-size: 0.65rem; font-weight: 600; '
        'letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; '
        'font-family: Inter, sans-serif;">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        label="nav",
        options=["📊 Sales Dashboard", "🔮 Churn Prediction", "🧪 Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color: #222222; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Author card
    st.markdown(
        '<div class="author-card">'
        '<div class="author-name">Soumendra Brahmapada</div>'
        '<div class="author-role">Data Analyst & ML Engineer</div>'
        '<div class="author-links">'
        '<a href="https://github.com/soumendra-brahmapada" target="_blank">⚡ GitHub</a>'
        '<a href="https://linkedin.com/in/soumendra-brahmapada" target="_blank">🔗 LinkedIn</a>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def page_sales_dashboard():
    """Render the Sales Dashboard page with KPIs and interactive charts."""

    # Hero banner
    st.markdown(
        '<div style="animation: fadeIn 0.6s ease-out;">'
        '<h1 style="color: #ededed; font-size: 1.8rem; font-weight: 700; '
        'font-family: Inter, sans-serif; margin-bottom: 0;">◈ EcomIQ Sales Intelligence</h1>'
        '<p style="color: #888888; font-size: 0.85rem; font-family: Inter, sans-serif; '
        'margin-top: 4px;">Real-time insights from 100K+ Olist transactions</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Analysing data..."):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.info("Please ensure all 9 Olist CSV files are in the data/raw/ directory.")
            return

    # ── KPI Section ─────────────────────────────────────────────────────
    render_section_header("KEY METRICS")

    summary = get_data_summary(df)
    total_revenue = summary.get("total_revenue", 0)
    total_orders = summary.get("total_orders", 0)
    avg_order_value = summary.get("avg_order_value", 0)

    # Calculate churn rate
    try:
        features = load_feature_data()
        churn_rate = features["churn_label"].mean() * 100
    except Exception:
        churn_rate = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_kpi_card("Total Revenue", f"R$ {total_revenue:,.0f}", "Lifetime GMV"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_kpi_card("Total Orders", f"{total_orders:,}", "Unique orders"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_kpi_card("Avg Order Value", f"R$ {avg_order_value:,.2f}", "Per transaction"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            render_kpi_card("Churn Rate", f"{churn_rate:.1f}%", "Inactive > 90 days"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Revenue Trend ───────────────────────────────────────────────────
    render_section_header("REVENUE OVERVIEW")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_trend = create_monthly_revenue_trend(df)
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render revenue trend: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_cats = create_top_categories_chart(df)
            st.plotly_chart(fig_cats, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render categories: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Customer & Order Analysis ───────────────────────────────────────
    render_section_header("CUSTOMER & ORDER ANALYSIS")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_state = create_state_distribution(df)
            st.plotly_chart(fig_state, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render state distribution: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_status = create_order_status_distribution(df)
            st.plotly_chart(fig_status, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render order status: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Reviews & Payments ──────────────────────────────────────────────
    render_section_header("REVIEWS & PAYMENTS")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_review = create_review_distribution(df)
            st.plotly_chart(fig_review, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render reviews: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_payment = create_payment_breakdown(df)
            st.plotly_chart(fig_payment, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render payments: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Delivery Performance ────────────────────────────────────────────
    render_section_header("DELIVERY PERFORMANCE")

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    try:
        fig_delivery = create_delivery_performance(df)
        st.plotly_chart(fig_delivery, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Could not render delivery performance: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def page_churn_prediction():
    """Render the Churn Prediction page with upload, gauge, and SHAP."""

    st.markdown(
        '<div style="animation: fadeIn 0.6s ease-out;">'
        '<h1 style="color: #ededed; font-size: 1.8rem; font-weight: 700; '
        'font-family: Inter, sans-serif; margin-bottom: 0;">🔮 Churn Prediction</h1>'
        '<p style="color: #888888; font-size: 0.85rem; font-family: Inter, sans-serif; '
        'margin-top: 4px;">Predict customer churn risk with XGBoost + SHAP explainability</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading models..."):
        try:
            models, results, X_train, X_test, y_train, y_test, feature_cols = load_trained_models()
            xgb_model = models["XGBoost"]
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return

    render_section_header("INPUT CUSTOMER DATA")

    input_mode = st.radio(
        "Choose input method:",
        ["Use sample customer", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    customer_data = None

    if input_mode == "Use sample customer":
        # Let user pick a sample from test set
        st.markdown(
            '<p style="color: #a1a1aa; font-size: 0.85rem;">Select a sample customer from '
            'the test dataset to see their churn prediction.</p>',
            unsafe_allow_html=True,
        )

        sample_idx = st.selectbox(
            "Sample customer index",
            options=list(range(min(20, len(X_test)))),
            format_func=lambda x: f"Customer #{x + 1}",
        )
        customer_data = X_test.iloc[[sample_idx]]

    else:
        uploaded = st.file_uploader(
            "Upload a CSV with customer features",
            type=["csv"],
            help=f"Required columns: {', '.join(feature_cols)}",
        )
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
                available = [c for c in feature_cols if c in uploaded_df.columns]
                if len(available) < len(feature_cols):
                    missing = set(feature_cols) - set(available)
                    st.warning(f"Missing columns: {missing}. Using available columns.")
                customer_data = uploaded_df[available].head(1)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if customer_data is not None:
        # ── Prediction ──────────────────────────────────────────────────
        render_section_header("PREDICTION RESULT")

        try:
            proba = xgb_model.predict_proba(customer_data)[0][1]
        except Exception:
            proba = float(xgb_model.predict(customer_data)[0])

        col_gauge, col_info = st.columns([2, 1])

        with col_gauge:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_gauge = create_churn_gauge(proba)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown(
                f'<div class="kpi-card" style="margin-top: 1rem;">'
                f'<div class="kpi-label">CHURN PROBABILITY</div>'
                f'<div class="kpi-value">{proba * 100:.1f}%</div>'
                f'<div style="margin-top: 10px;">{render_risk_badge(proba)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Customer features display
            st.markdown(
                '<div class="kpi-card" style="margin-top: 1rem;">'
                '<div class="kpi-label">CUSTOMER PROFILE</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            for col_name in customer_data.columns:
                val = customer_data[col_name].values[0]
                if isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                st.markdown(
                    f'<div style="display: flex; justify-content: space-between; '
                    f'padding: 4px 0; border-bottom: 1px solid #111111;">'
                    f'<span style="color: #888888; font-size: 0.8rem;">{col_name}</span>'
                    f'<span style="color: #ededed; font-size: 0.8rem; font-weight: 500;">'
                    f'{val_str}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── SHAP Explanation ────────────────────────────────────────────
        render_section_header("SHAP EXPLAINABILITY")

        try:
            shap_values, explainer = generate_shap_values(
                xgb_model, customer_data, feature_names=list(customer_data.columns)
            )

            # Configure matplotlib for dark theme
            plt.rcParams.update({
                "figure.facecolor": "#000000",
                "axes.facecolor": "#000000",
                "text.color": "#ededed",
                "axes.labelcolor": "#ededed",
                "xtick.color": "#a1a1aa",
                "ytick.color": "#a1a1aa",
                "axes.edgecolor": "#222222",
            })

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            fig_shap, ax = plt.subplots(figsize=(10, 5))

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                sv = shap_values
            else:
                sv = shap.Explanation(
                    values=shap_values[0] if isinstance(shap_values, list) else shap_values,
                    base_values=explainer.expected_value if not isinstance(
                        explainer.expected_value, np.ndarray
                    ) else explainer.expected_value[1],
                    data=customer_data.values[0],
                    feature_names=list(customer_data.columns),
                )

            shap.waterfall_plot(sv[0] if hasattr(sv, '__getitem__') else sv, show=False)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP waterfall plot unavailable: {e}")

        # ── Top Risk Factors ────────────────────────────────────────────
        render_section_header("TOP 5 RISK FACTORS")

        try:
            if hasattr(shap_values, "values"):
                sv_arr = shap_values.values[0]
            elif isinstance(shap_values, list):
                sv_arr = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
            else:
                sv_arr = shap_values[0]

            feat_importance = pd.DataFrame({
                "Feature": list(customer_data.columns),
                "SHAP Value": sv_arr,
                "Abs Impact": np.abs(sv_arr),
            }).sort_values("Abs Impact", ascending=False).head(5)

            rows_html = ""
            for _, row in feat_importance.iterrows():
                color = "#EF4444" if row["SHAP Value"] > 0 else "#0070f3"
                direction = "▲ Increases" if row["SHAP Value"] > 0 else "▼ Decreases"
                rows_html += (
                    f"<tr>"
                    f"<td style='color: #ededed; font-weight: 500;'>{row['Feature']}</td>"
                    f"<td style='color: {color};'>{row['SHAP Value']:.4f}</td>"
                    f"<td style='color: {color}; font-size: 0.8rem;'>{direction} churn risk</td>"
                    f"</tr>"
                )

            st.markdown(
                '<table class="styled-table">'
                "<thead><tr><th>Feature</th><th>SHAP Impact</th><th>Direction</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Could not compute risk factors: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def page_model_performance():
    """Render the Model Performance page with comparison, ROC, confusion matrix."""

    st.markdown(
        '<div style="animation: fadeIn 0.6s ease-out;">'
        '<h1 style="color: #ededed; font-size: 1.8rem; font-weight: 700; '
        'font-family: Inter, sans-serif; margin-bottom: 0;">🧪 Model Performance</h1>'
        '<p style="color: #888888; font-size: 0.85rem; font-family: Inter, sans-serif; '
        'margin-top: 4px;">Compare Logistic Regression, Random Forest, and XGBoost</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model results..."):
        try:
            models, results, X_train, X_test, y_train, y_test, feature_cols = load_trained_models()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return

    # ── Model Comparison Table ──────────────────────────────────────────
    render_section_header("MODEL COMPARISON")

    st.markdown(
        '<div class="chart-container">'
        + render_model_comparison_table(results)
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── ROC Curves ──────────────────────────────────────────────────────
    render_section_header("ROC CURVES")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            fig_roc = create_roc_curves(models, X_test, y_test)
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"ROC curves unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Confusion Matrix ────────────────────────────────────────────────
    with col_cm:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        try:
            y_pred = models["XGBoost"].predict(X_test)
            fig_cm = create_confusion_matrix_chart(y_test, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Confusion matrix unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature Importance ──────────────────────────────────────────────
    render_section_header("FEATURE IMPORTANCE")

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    try:
        fig_fi = create_feature_importance(models["XGBoost"], feature_cols)
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP Summary Plot ───────────────────────────────────────────────
    render_section_header("SHAP SUMMARY PLOT")

    try:
        plt.rcParams.update({
            "figure.facecolor": "#000000",
            "axes.facecolor": "#000000",
            "text.color": "#ededed",
            "axes.labelcolor": "#ededed",
            "xtick.color": "#a1a1aa",
            "ytick.color": "#a1a1aa",
            "axes.edgecolor": "#222222",
        })

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        shap_vals, explainer = generate_shap_values(
            models["XGBoost"], X_test, feature_names=feature_cols
        )

        fig_summary, ax = plt.subplots(figsize=(10, 5))

        if hasattr(shap_vals, "values"):
            shap_data = shap_vals.values
        elif isinstance(shap_vals, list):
            shap_data = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_data = shap_vals

        shap.summary_plot(
            shap_data, X_test, feature_names=feature_cols, show=False, plot_size=None
        )
        st.pyplot(fig_summary, use_container_width=True)
        plt.close(fig_summary)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"SHAP summary plot unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════

if page == "📊 Sales Dashboard":
    page_sales_dashboard()
elif page == "🔮 Churn Prediction":
    page_churn_prediction()
elif page == "🧪 Model Performance":
    page_model_performance()
