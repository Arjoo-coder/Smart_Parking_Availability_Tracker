"""Page 2 - Parking Availability: every location with a clear status indicator."""

import html

import pandas as pd
import streamlit as st

from app import ui
from src import dashboard_charts as charts

SORT_OPTIONS = {
    "Location name": ("location", True),
    "Most available spaces": ("available_spaces", False),
    "Highest occupancy": ("occupancy_pct", False),
    "Lowest occupancy": ("occupancy_pct", True),
}


def render(ctx):
    ui.page_header("Parking Availability",
                   f"Data as of {ctx.snapshot_ts:%A, %d %B %Y, %H:%M}")
    ui.status_legend()
    st.write("")

    # ---- filters ---------------------------------------------------------
    col_filter, col_sort = st.columns([2, 1])
    wanted = col_filter.multiselect("Show locations with status", list(ui.STATUS_ICONS),
                                    default=list(ui.STATUS_ICONS), key="av_status_filter")
    sort_label = col_sort.selectbox("Sort by", list(SORT_OPTIONS), key="av_sort")

    sort_column, ascending = SORT_OPTIONS[sort_label]
    shown = (ctx.snapshot[ctx.snapshot["status"].isin(wanted)]
             .sort_values(sort_column, ascending=ascending).reset_index(drop=True))

    if shown.empty:
        st.info("No location matches the selected status filter at this time. "
                "Add more statuses above, or pick a different snapshot in the sidebar.")
        return

    # ---- one card per location --------------------------------------------
    columns = st.columns(2, gap="medium")
    for i, row in enumerate(shown.itertuples()):
        with columns[i % 2]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="loc-row" style="border-bottom:none;padding-top:0">'
                    f'<div class="loc-name" style="font-size:1.1rem">{html.escape(row.location)}</div>'
                    f'<div>{ui.badge(row.status)}</div></div>'
                    f'<div class="big-number">{row.available_spaces} '
                    f'<span class="muted">spaces available</span></div>',
                    unsafe_allow_html=True)
                st.progress(float(min(max(row.occupancy_rate, 0.0), 1.0)))
                events = "event nearby" if row.special_event else "no event"
                st.caption(f"{row.occupied_spaces} occupied of {row.total_spaces} • "
                           f"{row.occupancy_pct:.0f}% full • {row.weather} • {events}")

    # ---- comparison chart + table -----------------------------------------
    st.subheader("Comparison")
    chart_col, table_col = st.columns([3, 2], gap="large")
    with chart_col:
        ui.show_fig(charts.fig_snapshot_stacked(shown))
    with table_col:
        table = pd.DataFrame({
            "Location": shown["location"],
            "Status": [f"{ui.STATUS_ICONS[s]} {s}" for s in shown["status"]],
            "Available": shown["available_spaces"],
            "Occupied": shown["occupied_spaces"],
            "Total": shown["total_spaces"],
            "Occupancy %": shown["occupancy_pct"].round(1),
        })
        st.dataframe(table, hide_index=True)
        st.caption("Status rule: " + " • ".join(
            f"{s}: {ui.STATUS_DESCRIPTIONS[s]}" for s in ui.STATUS_ICONS))
