"""Page 1 - Home Dashboard: headline numbers for the selected snapshot."""

import html

import streamlit as st

from app import ui
from src import dashboard_charts as charts
from src import dashboard_data as dd


def _change(current, previous, unit="", decimals=0):
    """Small text such as '▲ 102 vs 1 h earlier' (empty if there is no earlier reading)."""
    if previous is None:
        return ""
    diff = current - previous
    if round(diff, decimals) == 0:
        return "no change vs 1 h earlier"
    arrow = "▲" if diff > 0 else "▼"
    return f"{arrow} {abs(diff):,.{decimals}f}{unit} vs 1 h earlier"


def render(ctx):
    snapshot = ctx.snapshot
    summary = dd.overall_summary(snapshot)
    earlier = dd.overall_summary(ctx.previous_snapshot) if ctx.previous_snapshot is not None else None
    prev = lambda key: earlier[key] if earlier else None   # noqa: E731

    ui.hero("Smart Parking Dashboard",
            f"Snapshot of {ctx.snapshot_ts:%A, %d %B %Y at %H:%M} • change the moment in the sidebar")
    ui.simulated_notice()

    counts = snapshot["status"].value_counts()
    status_text = " • ".join(f"{counts.get(s, 0)} {s.lower()}" for s in ui.STATUS_ICONS)
    ui.kpi_row([
        {"label": "Total parking spaces", "value": f"{summary['total_spaces']:,}",
         "sub": f"across {summary['num_locations']} locations", "accent": "#264653"},
        {"label": "Currently occupied", "value": f"{summary['occupied_spaces']:,}",
         "sub": _change(summary["occupied_spaces"], prev("occupied_spaces")), "accent": "#e76f51"},
        {"label": "Currently available", "value": f"{summary['available_spaces']:,}",
         "sub": _change(summary["available_spaces"], prev("available_spaces")), "accent": "#2e9e5b"},
        {"label": "Overall occupancy", "value": f"{summary['occupancy_pct']:.1f}%",
         "sub": _change(summary["occupancy_pct"], prev("occupancy_pct"), " pts", 1), "accent": "#e0a100"},
        {"label": "Parking locations", "value": f"{summary['num_locations']}",
         "sub": status_text, "accent": "#2a9d8f"},
    ])

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Location status")
        rows = []
        for row in snapshot.itertuples():
            rows.append(
                f'<div class="loc-row"><div><div class="loc-name">{html.escape(row.location)}</div>'
                f'<div class="muted">{row.available_spaces} free of {row.total_spaces}'
                f' • {row.occupancy_pct:.0f}% full</div></div>'
                f'<div>{ui.badge(row.status)}</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)
        ui.status_legend()
    with right:
        st.subheader("Last 24 hours")
        ui.show_fig(charts.fig_last_24_hours(ctx.df, ctx.snapshot_ts))

    st.subheader("Quick insights")
    busiest = snapshot.sort_values("occupancy_pct", ascending=False).iloc[0]
    roomiest = snapshot.sort_values("available_spaces", ascending=False).iloc[0]
    st.markdown(
        f"- **{busiest['location']}** is the busiest location: {busiest['occupancy_pct']:.0f}% full "
        f"({busiest['status']}).\n"
        f"- **{roomiest['location']}** has the most free spaces: {roomiest['available_spaces']} "
        f"of {roomiest['total_spaces']}.\n"
        f"- Weather at this time: **{snapshot['weather'].mode().iloc[0]}**.")
    events = snapshot[snapshot["special_event"] == 1]["location"].tolist()
    if events:
        st.markdown(f"- Special event near: **{', '.join(events)}**.")

    st.info("Next steps: open **Parking Availability** for every location in detail, "
            "**Predict Availability** to forecast a future time, or **Charts & Analytics** "
            "to explore the patterns.")
