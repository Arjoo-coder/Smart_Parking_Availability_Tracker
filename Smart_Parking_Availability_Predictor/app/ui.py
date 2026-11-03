"""Shared look-and-feel helpers for the dashboard: CSS, KPI cards, status badges."""

import html

import matplotlib.pyplot as plt
import streamlit as st

from src import config

STATUS_ICONS = {"Available": "\U0001F7E2", "Limited": "\U0001F7E1", "Full": "\U0001F534"}

STATUS_DESCRIPTIONS = {
    "Available": f"under {config.STATUS_LIMITED_FROM:.0%} full",
    "Limited": f"{config.STATUS_LIMITED_FROM:.0%} to under {config.STATUS_FULL_FROM:.0%} full",
    "Full": f"{config.STATUS_FULL_FROM:.0%} full or more",
}

STATUS_ACCENT = {"Available": "#2e9e5b", "Limited": "#e0a100", "Full": "#d64545"}

# NOTE: keep this block free of blank lines - Streamlit's markdown parser
# would otherwise treat the part after a blank line as normal text.
CSS = """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #1f6f78 0%, #2a9d8f 60%, #58c4b6 100%); color: #ffffff; padding: 22px 26px; border-radius: 16px; margin-bottom: 14px;}
.hero-title {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.hero-sub {margin-top: 4px; opacity: 0.92; font-size: 1rem;}
.kpi-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 14px 16px; background: rgba(42,157,143,0.07); min-height: 118px; margin-bottom: 8px;}
.kpi-label {font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; font-weight: 600;}
.kpi-value {font-size: 1.95rem; font-weight: 700; line-height: 1.25; margin: 2px 0;}
.kpi-sub {font-size: 0.8rem; opacity: 0.7;}
.badge {display: inline-block; padding: 2px 12px; border-radius: 999px; color: #ffffff; font-weight: 600; font-size: 0.8rem; white-space: nowrap;}
.badge-available {background: #2e9e5b;}
.badge-limited {background: #e0a100;}
.badge-full {background: #d64545;}
.pill-real {display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; background: #d8f3e4; color: #14633a; border: 1px solid #9bd9b7;}
.pill-sim {display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; background: #fff3cd; color: #7a5b00; border: 1px solid #f0d77a;}
.loc-row {display: flex; justify-content: space-between; align-items: center; padding: 9px 2px; border-bottom: 1px solid rgba(128,128,128,0.2);}
.loc-name {font-weight: 600;}
.muted {opacity: 0.65; font-size: 0.85rem;}
.big-number {font-size: 2.1rem; font-weight: 700; line-height: 1.2;}
@media (max-width: 640px) {.hero-title {font-size: 1.4rem;} .kpi-value {font-size: 1.6rem;} .kpi-card {min-height: 0;}}
</style>
"""


# Whether the data on screen is simulated is decided by the saved dataset (data_source.json),
# and set once per run by streamlit_app.py. It defaults to "simulated" to be on the safe side.
_STATE = {"simulated": True}


def set_data_kind(is_simulated):
    _STATE["simulated"] = bool(is_simulated)


def is_simulated():
    return _STATE["simulated"]


def data_pill():
    """HTML badge: SIMULATED DATA or REAL DATA."""
    if _STATE["simulated"]:
        return '<span class="pill-sim">SIMULATED DATA</span>'
    return '<span class="pill-real">REAL DATA (supplied)</span>'


def estimate_notice():
    """Reminder shown wherever a prediction is displayed."""
    st.caption("\u26A0\uFE0F Predictions are **estimates** from a statistical model. They are not "
               "guaranteed or real-time availability.")


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-title">{html.escape(title)}</div>'
                f'<div class="hero-sub">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.title(title)
    st.markdown(f'{data_pill()} &nbsp;<span class="muted">{html.escape(subtitle)}</span>',
                unsafe_allow_html=True)
    st.write("")


def simulated_notice():
    if _STATE["simulated"]:
        st.info("**Simulated data.** Every number in this dashboard comes from artificial, "
                "rule-generated parking data (not real sensors). It demonstrates the machine "
                "learning workflow; it is not a live parking feed.")
    else:
        st.info("**Supplied data.** The numbers come from the parking dataset you provided, not from a "
                "live sensor feed. Predictions are estimates based on that history.")


def badge(status):
    return f'<span class="badge badge-{status.lower()}">{html.escape(status)}</span>'


def status_legend():
    """One line explaining the three status labels."""
    parts = [f'{badge(s)} <span class="muted">{STATUS_DESCRIPTIONS[s]}</span>' for s in STATUS_ICONS]
    st.markdown(" &nbsp;&nbsp; ".join(parts), unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent=None):
    style = f' style="border-left: 5px solid {accent};"' if accent else ""
    return (f'<div class="kpi-card"{style}><div class="kpi-label">{html.escape(str(label))}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    """Show a row of KPI cards. `items` = list of dicts: label, value, sub, accent."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.markdown(kpi_card(item["label"], item["value"], item.get("sub", ""),
                                 item.get("accent")), unsafe_allow_html=True)


def show_fig(fig):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)
