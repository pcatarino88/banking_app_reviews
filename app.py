from __future__ import annotations

from datetime import datetime
import io
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

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
        padding-top: 1.5rem;
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

# === CONFIG FOR TAB 2 ===========================

@st.cache_data(show_spinner=False)
def load_tab2_frame() -> pd.DataFrame:
    cols = [APP_COL, DATE_COL, SCORE_COL, TOPIC_POS_COL, TOPIC_NEG_COL]
    url = st.secrets.get("DATA_TAB2", None)
    if url:
        df = pd.read_parquet(url, columns=cols)  # <-- select columns!
    else:
        df = pd.read_parquet("assets/df_final.parquet", columns=cols)

    # Light, memory-friendly dtypes
    df[DATE_COL]  = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce").astype("Int64")

    # Categories reduce memory & speed up groupby
    for c in (APP_COL, TOPIC_POS_COL, TOPIC_NEG_COL):
        df[c] = df[c].astype("category")

    return df

APP_COL = "app"                
DATE_COL = "review_date"        
SCORE_COL = "score"            
TOPIC_POS_COL = "topic_label_POS"
TOPIC_NEG_COL = "topic_label_NEG"
UNDEFINED_LABEL = "Undefined"

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# -------------------------------
# III. Load dataframes from HF
# -------------------------------
df_tab1 = load_monthly_from_hf()

df_tab2 = load_tab2_frame()

# -------------------------------
# Main
# -------------------------------
def main():
    show_memory_usage()

    st.title("📱 Banking App Reviews")
    st.caption("Interactive analysis of app store ratings and reviews.")

    # Tabs
    app_tab, topics_tab, reviews_tab = st.tabs(
        ["App Ratings", "Key Topics", "Search Reviews (WiP)"]
    )
    
    # -------------------------------
    # TAB 1: APP RATINGS (monthly-based)
    # -------------------------------
    with app_tab:
        st.subheader("Evolution of Apps' rating")

        with st.container():
            cols = st.columns([2, 2, 2])

            # 1) Bank App filter
            app_list = sorted(df_tab1["app"].dropna().unique().tolist())
            selected_apps = cols[0].multiselect(
                "Bank App", options=app_list, default=app_list,
                help="Choose one or more apps/banks."
            )

            # 2) Time Period slider 
            min_date = df_tab1["period_month"].min()
            max_date = df_tab1["period_month"].max()
            default_start = max(min_date, max_date - pd.DateOffset(years=4))
            start_date, end_date = cols[1].slider(
                "Time Period",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD",
                help="Filter by review date."
            )

            # 3) Time Unit 
            unit = cols[2].selectbox(
                "Time Unit",
                options=["Month", "Quarter", "Semester", "Year"],
                index=1,
                help="Select preferred time unit: Month, Quarter, Semester or Year."
            )

        # Apply filters on the monthly DF
        mask = (
            df_tab1["period_month"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            & (df_tab1["app"].isin(selected_apps) if selected_apps else True)
        )
        df_f = df_tab1.loc[mask].copy()

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

        st.markdown("---")
        st.markdown(
            """
            <small>
            <strong>Data source:</strong> Google Play Store reviews. <strong>Last Update:</strong> 4th September 2025.<br/>
            <strong>Notes:</strong> Ratings for each time unit are simple averages of the monthly averages (not weighted by review counts).
            </small>
            """,
            unsafe_allow_html=True,
        )
    
    # -------------------------------
    # TAB 2: TOPIC MODELING
    # -------------------------------
    
    with topics_tab:
        st.subheader("Key topics mentioned in reviews")
    
        # --- Filters row ---------------------------
        with st.container():
            cols = st.columns([2, 2, 2])

            # i) Bank App select
            apps = sorted(df_tab2[APP_COL].dropna().unique().tolist())
            sel_apps = cols[0].multiselect(
                "Bank App",
                options=apps,
                default=apps,
                help="Choose one or more apps/banks."
            )
        
            # ii) Time Period slider
            min_dt = pd.to_datetime(df_tab2[DATE_COL].min())
            max_dt = pd.to_datetime(df_tab2[DATE_COL].max())
            default_start = max(min_dt, max_dt - pd.DateOffset(years=4))
            start_dt, end_dt = cols[1].slider(
                "Time Period",
                min_value=min_dt.to_pydatetime(),
                max_value=max_dt.to_pydatetime(),
                value=(default_start.to_pydatetime(), max_dt.to_pydatetime()),
                format="YYYY-MM-DD",
                help="Filter by review date."
            )
    
            # iii) Type of reviews
            review_type = cols[2].radio(
                "Type of reviews",
                options=["Positive", "Negative"],
                index=1,           # default Negative
                horizontal=True,
                help="Select either 'Positive' or 'Negative' reviews."
            )
        
        # Decide which topic column + scores to use
        topic_col = TOPIC_POS_COL if review_type == "Positive" else TOPIC_NEG_COL
        valid_scores = {4, 5} if review_type == "Positive" else {1, 2}
    
        # --- Lightweight filtering ------------------------------
        mask = (
            df_tab2[APP_COL].isin(sel_apps)
            & df_tab2[SCORE_COL].isin(valid_scores)
            & df_tab2[DATE_COL].between(pd.Timestamp(start_dt), pd.Timestamp(end_dt))
            & df_tab2[topic_col].notna()
            & (df_tab2[topic_col] != UNDEFINED_LABEL)
        )
        
        view = df_tab2.loc[mask, [APP_COL, topic_col]]
        if view.empty:
            st.info("No reviews match the current filters.")
            st.stop()
    
        # --- Prepare topic order & colors -----------------------------------------
        # Keep a stable topic order: use our color dict order where available, then any extras
        topics_in_view = view[topic_col].cat.remove_unused_categories().cat.categories.tolist()
        ordered_topics = topics_in_view

        # Build a deterministic color map for the topics we actually have
        color_map = {t: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, t in enumerate(ordered_topics)}
        
        # Aggregate to proportions per app
        ct = (view.groupby([APP_COL, topic_col], observed=True)
                  .size()
                  .rename("n")
                  .reset_index())
    
        # Ensure every (app, topic) pair exists -> aligned stacks
        all_index = pd.MultiIndex.from_product([sel_apps, ordered_topics], names=[APP_COL, topic_col])
        ct = ct.set_index([APP_COL, topic_col]).reindex(all_index, fill_value=0).reset_index()
        
        totals = ct.groupby(APP_COL, as_index=False)["n"].sum().rename(columns={"n":"total_n"})
        ct = ct.merge(totals, on=APP_COL, how="left")
        ct["pct"] = np.where(ct["total_n"]>0, ct["n"]/ct["total_n"]*100.0, 0.0)
        
        x_order = sel_apps
        order_map = {a:i for i,a in enumerate(x_order)}
    
        # --- Build figure (stacked 100%) ------------------------------------------
        fig = go.Figure()
        for topic in ordered_topics:
            df_t = ct[ct[topic_col]==topic].sort_values(APP_COL, key=lambda s: s.map(order_map))
            fig.add_trace(
                go.Bar(
                    x=df_t[APP_COL],
                    y=df_t["pct"],
                    name=topic,
                    marker_color=color_map[topic],
                    text=(df_t["pct"].round().astype(int).astype(str) + "%").where(df_t["pct"]>=4, ""),
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        + topic + ": %{y:.1f}%<br>"
                        + "Count: %{customdata}"
                        + "<extra></extra>"
                    ),
                    customdata=df_t["n"],
                )
            )
        
        for app, total in totals.set_index(APP_COL).reindex(x_order)["total_n"].items():
            fig.add_annotation(x=app, y=100, yshift=16, text=f"n={int(total)}", showarrow=False, font=dict(size=11))
        
        fig.update_layout(
            barmode="stack",
            yaxis=dict(title="Proportion of reviews", range=[0, 100], ticksuffix="%", showgrid=True),
            xaxis=dict(title="App / Bank"),
            legend=dict(title="Topic", orientation="v", x=1.02, y=1.0, bgcolor="rgba(255,255,255,0.15)"),
            margin=dict(l=20, r=160, t=30, b=60),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
    
        # Footnote / notes
        st.caption(
            "Notes: Proportions are within each app (stacked to 100%). "
            "‘Positive’ uses scores 4–5 and ‘Negative’ uses scores 1–2. "
            "Reviews that were not possible to allocate to a specific topic were removed from this analysis."
        )  
    
    # -------------------------------
    # Tab 3 placeholder
    # -------------------------------

    with reviews_tab:
        st.subheader("Deep dive on real reviews")
        st.info("🚧 Work in progress")


if __name__ == "__main__":
    main()
