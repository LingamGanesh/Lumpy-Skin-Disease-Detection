"""Database-backed analytics dashboard for LumpyCare."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from lumpycare.database import (
    analytics_activity, analytics_alerts, analytics_distribution, analytics_geo,
    analytics_stats, analytics_trends,
)

GREEN = "#16815f"
RED = "#df625d"
SLATE = "#71847b"


def _metric(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="muted">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    st.markdown("### Herd screening overview")
    st.caption("Analytics are calculated from saved screening records. They are screening indicators, not confirmed diagnoses.")

    filter_cols = st.columns([1, 1, 1.25, 1.1])
    with filter_cols[0]:
        period = st.selectbox("Date range", ["7 days", "30 days", "90 days", "All time"], index=1)
    with filter_cols[1]:
        result_type = st.selectbox("Result", ["All results", "Lumpy skin", "Normal skin"])
    with filter_cols[2]:
        confidence_pct = st.slider("Confidence range", 0, 100, (0, 100), step=5)
    with filter_cols[3]:
        include_demo = st.checkbox("Include demo records", value=True)

    now = datetime.now(timezone.utc)
    start_date = None
    if period != "All time":
        days = {"7 days": 7, "30 days": 30, "90 days": 90}[period]
        start_date = (now - timedelta(days=days)).date().isoformat()
    filters = {
        "start_date": start_date,
        "end_date": now.date().isoformat(),
        "result_type": None if result_type == "All results" else result_type,
        "min_confidence": confidence_pct[0] / 100,
        "max_confidence": confidence_pct[1] / 100,
        "include_demo": include_demo,
    }

    stats = analytics_stats(**filters)
    trends = analytics_trends(**filters)
    distribution = analytics_distribution(**filters)
    alerts = analytics_alerts(threshold=.30, **filters)
    activity = analytics_activity(limit=8, **filters)
    geo = analytics_geo(**filters)

    if stats["demo_scans"]:
        st.info(
            f"Demo mode: {stats['demo_scans']} of the records in this view are synthetic examples. "
            "They are not real farm observations or confirmed outbreak cases."
        )

    if stats["total_scans"] == 0:
        st.info("No saved scans match these filters. Run a screening or include the synthetic demo records to explore the dashboard.")
    metric_cols = st.columns(4)
    cards = [
        ("Total scans", f'{stats["total_scans"]:,}', "Saved records in selected period"),
        ("Infected predictions", f'{stats["infected"]:,}', "Model predicted Lumpy skin"),
        ("Healthy predictions", f'{stats["healthy"]:,}', "Model predicted Normal skin"),
        ("Average confidence", f'{stats["average_confidence"] * 100:.1f}%', "Model probability estimate"),
    ]
    for col, (label, value, note) in zip(metric_cols, cards):
        with col:
            _metric(label, value, note)

    secondary = st.columns(3)
    with secondary[0]:
        st.metric("Infection prediction rate", f'{stats["infection_rate"] * 100:.1f}%')
    with secondary[1]:
        st.metric("Scans today", f'{stats["scans_today"]:,}')
    with secondary[2]:
        st.metric("Scans in last 7 days", f'{stats["scans_week"]:,}')

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Infection trend")
        if trends:
            frame = pd.DataFrame(trends)
            frame["day"] = pd.to_datetime(frame["day"])
            fig, ax = plt.subplots(figsize=(8, 3.3))
            ax.plot(frame["day"], frame["infected"], color=RED, linewidth=2.5, marker="o", markersize=3, label="Lumpy skin")
            ax.plot(frame["day"], frame["healthy"], color=GREEN, linewidth=2.5, marker="o", markersize=3, label="Normal skin")
            ax.fill_between(frame["day"], frame["infected"], color=RED, alpha=.10)
            ax.set_ylabel("Scans")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.legend(frameon=False, ncol=2, loc="upper left")
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.caption("Trend chart will appear when matching scan records are available.")
    with right:
        st.markdown("#### Healthy vs infected predictions")
        if stats["total_scans"]:
            fig, ax = plt.subplots(figsize=(5, 3.3))
            values = [stats["infected"], stats["healthy"]]
            ax.pie(
                values, startangle=90, counterclock=False, colors=[RED, GREEN],
                wedgeprops={"width": .34, "edgecolor": "white", "linewidth": 2},
            )
            ax.text(0, 0, f'{stats["total_scans"]:,}\nscans', ha="center", va="center",
                    fontsize=17, fontweight="bold", color="#234237")
            ax.legend(["Lumpy skin", "Normal skin"], loc="lower center",
                      bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
            ax.set(aspect="equal")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.caption("Prediction distribution will appear after the first saved scan.")

    st.write("")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Confidence distribution")
        confidence_rows = distribution["confidence"]
        if confidence_rows:
            labels = [row["bucket"] for row in confidence_rows]
            counts = [row["count"] for row in confidence_rows]
            fig, ax = plt.subplots(figsize=(6.4, 3.1))
            ax.bar(labels, counts, color=GREEN, width=.68)
            ax.set_ylabel("Scans")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.caption("No confidence data for the selected filters.")
    with chart_right:
        st.markdown("#### Top affected demo / recorded locations")
        if geo:
            geo_frame = pd.DataFrame(geo).sort_values("infected", ascending=True).tail(6)
            fig, ax = plt.subplots(figsize=(6.4, 3.1))
            ax.barh(geo_frame["location"], geo_frame["infected"], color=RED, alpha=.85)
            ax.set_xlabel("Lumpy skin predictions")
            ax.grid(axis="x", alpha=.2)
            ax.spines[["top", "right", "left"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.caption("Location analytics need a scan linked to a village or GPS coordinate.")

    st.write("")
    alert_col, activity_col = st.columns([1, 1.35])
    with alert_col:
        st.markdown("#### Outbreak signal alerts")
        st.caption("Illustrative screening threshold: 30% predicted positive rate, minimum 5 scans per location.")
        if alerts:
            for alert in alerts[:6]:
                rate = float(alert["infection_rate"]) * 100
                label = "Synthetic demo signal" if alert["demo_scans"] else "Recorded screening signal"
                st.warning(f'**{alert["location"]}** — {rate:.1f}% positive ({alert["infected"]}/{alert["scans"]}). {label}. Vet review required.')
        else:
            st.success("No location-level screening signals cross the configured threshold.")
    with activity_col:
        st.markdown("#### Recent scans")
        if activity:
            table = pd.DataFrame(activity)
            columns = ["scanned_at", "image_name", "prediction", "confidence", "cattle_name", "location", "is_demo"]
            table = table[[name for name in columns if name in table.columns]].rename(columns={
                "scanned_at": "Scanned at (UTC)", "image_name": "Image",
                "prediction": "Prediction", "confidence": "Confidence",
                "cattle_name": "Cattle", "location": "Location", "is_demo": "Demo",
            })
            if "Confidence" in table:
                table["Confidence"] = (table["Confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.caption("Recent saved scans will appear here.")

    st.caption("Important: map-level clusters and disease severity are not inferred from confidence. A qualified veterinarian must assess suspected cases.")
