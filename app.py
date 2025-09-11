from __future__ import annotations

from datetime import datetime
import io
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from huggingface_hub import hf_hub_download  # NEW

import psutil  # memory widget

# -------------------------------
# I. Page config
# -------------------------------
st.set_page_config(
    page_title="Banking App Reviews",
    page_icon="📱",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* reduce the big top padding Streamlit applies */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def show_memory_usage():
    p = psutil.Process(os.getpid())
    st.sidebar.caption(f"RAM used: {p.memory_info().rss / 1024**2:.0f} MB")

# -------------------------------
# II. Helpers
# -------------------------------

def clean_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure monthly schema + dtypes."""
    expected = {"period_month", "app", "avg_score", "n_reviews"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["period_month"] = pd.to_datetime(df["period_month"], errors="coerce")
    df = df.dropna(subset=["period_month"])

    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")
    df["n_reviews"] = pd.to_numeric(df["n_reviews"], errors="coerce").fillna(0).astype("int32")

    if df["app"].dtype == object:
        df["app"] = df["app"].astype("category")
    return df

@st.cache_data(show_spinner="Loading monthly data…")
def load_monthly_from_hf() -> pd.DataFrame:
    """
    Preferred path: pull df_monthly.parquet from a HF dataset repo.
    Required secrets:
      - HF_REPO_ID = "pcatarino88/banking-app-reviews-data"
      - HF_FILE_DF_MONTHLY = "df_monthly.parquet" (optional; defaults to that)
      - HF_TOKEN (only if the repo is private)
    Fallback (public raw URL): DATA_URL_DF_MONTHLY
    """
    repo_id = st.secrets.get("HF_REPO_ID")
    filename = st.secrets.get("HF_FILE_DF_MONTHLY", "df_monthly.parquet")
    token = st.secrets.get("HF_TOKEN")

    if repo_id:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
        return clean_monthly(pd.read_parquet(local_path))

    # Fallback to a direct URL (public)
    url = st.secrets.get("DATA_URL_DF_MONTHLY")
    if url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return clean_monthly(pd.read_parquet(io.BytesIO(r.content), engine="pyarrow"))

    raise RuntimeError("No HF_REPO_ID/filename or DATA_URL_DF_MONTHLY provided in secrets.")

def period_start_from_unit(dt: pd.Series, unit: str) -> pd.Series:
    """(Kept for semantic parity) Convert timestamps to the start of Month/Quarter/Semester/Year."""
    dt = pd.to_datetime(dt)
    unit = unit.lower()
    if unit == "month":
        return dt.dt.to_period("M").dt.to_timestamp()
    if unit == "quarter":
        return dt.dt.to_period("Q").dt.start_time
    if unit == "semester":
        y = dt.dt.year
        h2 = dt.dt.month > 6
        return pd.to_datetime({"year": y, "month": np.where(h2, 7, 1), "day": 1})
    if unit == "year":
        return dt.dt.to_period("Y").dt.to_timestamp()
    return dt

@st.cache_data(show_spinner=False)
def rebin_from_month(df_monthly: pd.DataFrame, unit: str) -> pd.DataFrame:
    """
    Rebin already-monthly data to Month/Quarter/Semester/Year without using resample,
    to avoid MultiIndex concat issues on filtered categorical groups.
    """
    freq_map = {"Month": "M", "Quarter": "Q", "Semester": "2Q", "Year": "Y"}
    freq = freq_map[unit]

    df = df_monthly.copy()

    # make sure 'app' isn't categorical to avoid pandas concat bugs
    df["app"] = df["app"].astype(str)

    # build the target period and aggregate
    df["period"] = df["period_month"].dt.to_period(freq).dt.start_time

    out = (
        df.groupby(["app", "period"], observed=True)
          .agg(avg_score=("avg_score", "mean"),
               n_reviews=("n_reviews", "sum"))
          .reset_index()
          .sort_values(["period", "app"])
    )
    return out

# Brand colors (kept)
BRAND_COLORS = {
    "Barclays": "#00AEEF",
    "HSBC": "#FF4D4D",
    "Lloyds": "#005A2B",
    "Monzo": "#14233C",
    "Revolut": "#001A72",
    "Santander UK": "#EC0000",
    "Santander": "#EC0000",
}

DEFAULT_CYCLE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]

def build_brand_palette(apps: list[str]) -> dict[str, str]:
    palette = {}
    idx = 0
    for app in apps:
        if app in BRAND_COLORS:
            palette[app] = BRAND_COLORS[app]
        else:
            palette[app] = DEFAULT_CYCLE[idx % len(DEFAULT_CYCLE)]
            idx += 1
    return palette

def palette_in_order(app_order: list[str], palette: dict[str, str]) -> list[str]:
    colors = []
    fallback_cycle = iter(DEFAULT_CYCLE)
    for app in app_order:
        colors.append(palette.get(app, next(fallback_cycle)))
    return colors

# -------------------------------
# III. Load monthly DF from HF
# -------------------------------
df = load_monthly_from_hf()

# -------------------------------
# Main
# -------------------------------
def main():
    show_memory_usage()

    st.title("📱 Banking App Reviews")
    st.caption("Interactive analysis of app store ratings and reviews.")

    # Tabs
    app_tab, topics_tab, reviews_tab = st.tabs(
        ["App Ratings", "Key Topics (WiP)", "Search Reviews (WiP)"]
    )
    
    # -------------------------------
    # TAB 1: APP RATINGS (monthly-based)
    # -------------------------------
    with app_tab:
        st.subheader("Evolution of Apps' rating")

        with st.container():
            cols = st.columns([2, 3, 2])

            # 1) Bank App filter
            app_list = sorted(df["app"].dropna().unique().tolist())
            selected_apps = cols[0].multiselect(
                "Bank App", options=app_list, default=app_list
            )

            # 2) Time Period slider (now based on period_month)
            min_date = df["period_month"].min()
            max_date = df["period_month"].max()
            default_start = max(min_date, max_date - pd.DateOffset(years=4))
            start_date, end_date = cols[1].slider(
                "Time Period",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD",
            )

            # 3) Time Unit (we rebin from monthly)
            unit = cols[2].selectbox(
                "Time Unit",
                options=["Month", "Quarter", "Semester", "Year"],
                index=1,
            )

        # Apply filters on the monthly DF
        mask = (
            df["period_month"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            & (df["app"].isin(selected_apps) if selected_apps else True)
        )
        df_f = df.loc[mask].copy()

        if df_f.empty:
            st.info("No data for the selected filters.")
            st.stop()

        # Rebin Month -> selected unit (keeps table small)
        agg = rebin_from_month(df_f, unit)

        # KPIs (use review counts from agg and weighted mean for rating)
        total_reviews = int(agg["n_reviews"].sum())
        apps_count = len(selected_apps)
        weighted_avg = (
            (agg["avg_score"] * agg["n_reviews"]).sum() / max(total_reviews, 1)
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Reviews (filtered)", f"{total_reviews:,}")
        k2.metric("Apps selected", f"{apps_count}")
        k3.metric("Avg. rating", f"{weighted_avg:.2f} / 5")
        
        # Chart (Altair)
        palette = build_brand_palette(sorted(df_f["app"].dropna().unique().tolist()))
        latest = agg.sort_values("period").groupby("app", as_index=False).tail(1)
        legend_order = latest.sort_values("avg_score", ascending=False)["app"].tolist()
        color_range = palette_in_order(legend_order, palette)

        base = alt.Chart(agg).mark_line(point=True).encode(
            x=alt.X(
                "period:T",
                title="Time",
                axis=alt.Axis(
                    format="%b/%y",        # e.g., Oct/24
                    labelAngle=0,          # optional: keep labels horizontal
                    labelOverlap=True
                ),
            ),
            y=alt.Y("avg_score:Q", title="Average rating"),
            color=alt.Color(
                "app:N",
                title="App",
                sort=legend_order,
                scale=alt.Scale(domain=legend_order, range=color_range),
                legend=alt.Legend(title="App",labelFontSize=12,titleFontSize=13,symbolSize=80),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="Period"),
                alt.Tooltip("app:N", title="App"),
                alt.Tooltip("avg_score:Q", title="Avg. rating", format=".2f"),
            ],
        ).properties(height=420)

        st.altair_chart(base, use_container_width=True)


    # -------------------------------
    # Tab 2 & 3 placeholders
    # -------------------------------
    with topics_tab:
        st.subheader("Find what are the main topics mentioned in reviews")
        st.info("🚧 Work in progress")

    with reviews_tab:
        st.subheader("Deep dive on real reviews")
        st.info("🚧 Work in progress")

    # -------------------------------
    # Footnote
    # ------------------------------- 
    st.markdown("---")
    st.markdown(
        """
        <small>
        <strong>Data source:</strong> Google Play Store reviews. Last update: 04/09/2025.<br/>
        <strong>Notes:</strong> Ratings are simple averages for each time period.
        </small>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
