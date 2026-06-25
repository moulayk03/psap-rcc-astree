"""
ui.py — Composants visuels réutilisables : CSS custom, en-têtes, cartes KPI.
"""
import streamlit as st


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---- Police & espacements généraux ---- */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* ---- Titres ---- */
        h1 {
            color: #1F3864;
            font-weight: 800 !important;
            border-bottom: 3px solid #2E75B6;
            padding-bottom: 0.4rem;
            margin-bottom: 1.2rem !important;
        }
        h2, h3 {
            color: #1F3864;
            font-weight: 700 !important;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1F3864 0%, #2E5090 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #F4F7FB !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-size: 0.95rem;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.2);
        }

        /* Radio buttons in sidebar -> pill style on hover */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 6px 10px;
            margin-bottom: 4px;
            transition: background-color 0.15s ease-in-out;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: rgba(255,255,255,0.18);
        }

        /* ---- Metrics (st.metric) ---- */
        div[data-testid="stMetric"] {
            background-color: #F4F7FB;
            border: 1px solid #DCE5F0;
            border-radius: 12px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 1px 3px rgba(31,56,100,0.07);
        }
        div[data-testid="stMetricLabel"] {
            color: #5B6B82 !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #1F3864 !important;
            font-weight: 800 !important;
        }

        /* ---- DataFrames ---- */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #DCE5F0;
        }

        /* ---- Tabs ---- */
        button[data-baseweb="tab"] {
            font-weight: 600;
        }

        /* ---- Expander ---- */
        details {
            background-color: #F4F7FB;
            border-radius: 10px;
            border: 1px solid #DCE5F0;
        }

        /* ---- Caption / small text ---- */
        .small-note {
            color: #5B6B82;
            font-size: 0.85rem;
        }

        /* ---- Badge pill ---- */
        .badge {
            display: inline-block;
            background-color: #DEEAF1;
            color: #1F3864;
            border-radius: 999px;
            padding: 3px 12px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", badges: list[str] | None = None):
    """En-tête de page standardisé avec badges optionnels."""
    st.title(title)
    if badges:
        st.markdown(
            " ".join(f'<span class="badge">{b}</span>' for b in badges),
            unsafe_allow_html=True,
        )
    if subtitle:
        st.markdown(f'<p class="small-note">{subtitle}</p>', unsafe_allow_html=True)
    st.write("")


def kpi_row(items: list[tuple[str, str]]):
    """Affiche une rangée de st.metric à partir d'une liste (label, valeur)."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
