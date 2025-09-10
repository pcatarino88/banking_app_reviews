from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import io, requests, streamlit as st
import altair as alt
from pathlib import Path

# -------------------------------
# I. Page config
# -------------------------------
st.set_page_config(
page_title="Banking App Reviews",
page_icon="📱",
layout="wide",
)

st.title("📱 Banking App Reviews")
st.caption("Interactive analysis of app store ratings and reviews.")

LOCAL = Path(__file__).parent / "assets" / "df_final.parquet"

# -------------------------------
# II. Helpers
# -------------------------------

@st.cache_data(show_spinner=True)
def load_local_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return clean_df(df)

@st.cache_data(show_spinner=True)
def load_parquet_from_url(url: str) -> pd.DataFrame:
    # cache on the *string URL*, do the download inside
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content), engine="pyarrow")
    return clean_df(df)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep required columns
    expected = {"review_id", "app", "score", "review_text", "review_date"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    # Types & cleaning (cheap but prevents repeated heavy work downstream)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int8")

    # Memory optimizations
    if df["app"].dtype == object:
        df["app"] = df["app"].astype("category")

    # If you don’t plot/use full text, consider:
    # df = df.drop(columns=["review_text"])

    return df
    
def period_start_from_unit(dt: pd.Series, unit: str) -> pd.Series:
    dt = pd.to_datetime(dt)
    unit = unit.lower()
    if unit == "month":
        return dt.dt.to_period("M").dt.to_timestamp()
    if unit == "quarter":
        return dt.dt.to_period("Q").dt.start_time
    if unit == "semester":
        # map to H1: Jan 1, H2: Jul 1
        y = dt.dt.year
        h2 = dt.dt.month > 6
        return pd.to_datetime({
            "year": y,
            "month": np.where(h2, 7, 1),
            "day": 1,
        })
    if unit == "year":
        return dt.dt.to_period("Y").dt.to_timestamp()
    return dt

@st.cache_data(show_spinner=False)
def aggregate_avg_rating(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    gcol = period_start_from_unit(df["review_date"], unit)
    tmp = df.assign(period=gcol)
    agg = (
        tmp.groupby(["period", "app"], as_index=False)["score"].mean()
        .rename(columns={"score": "avg_score"})
        .sort_values(["period", "app"]) # for stable plotting
    )
    return agg

# Brand color map (approximate official palettes). Extend as needed.
BRAND_COLORS = {
    "Barclays": "#00AEEF", # Barclays blue
    "HSBC": "#FF4D4D", # lighter red for distinction,
    "Lloyds": "#005A2B", # Lloyds green
    "Monzo": "#14233C", # Monzo navy
    "Revolut": "#001A72", # Revolut deep blue
    "Santander UK": "#EC0000", # Santander red
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
    """Return a color list aligned to a given domain order."""
    colors = []
    fallback_cycle = iter(DEFAULT_CYCLE)
    for app in app_order:
        colors.append(palette.get(app, next(fallback_cycle)))
    return colors

# -------------------------------
# III. Load
# -------------------------------

def get_df() -> pd.DataFrame:
    if LOCAL.exists():
        return load_local_parquet(str(LOCAL))

    url = st.secrets.get("DATA_URL")
    if url:
        return load_parquet_from_url(url)

    up = st.file_uploader("Upload df_final.parquet", type="parquet")
    if not up:
        st.stop()
    df = pd.read_parquet(up, engine="pyarrow")
    return clean_df(df)

df = get_df()

# -------------------------------
# Tabs
# -------------------------------
app_tab, topics_tab, reviews_tab = st.tabs(["App Ratings", "Key Topics (WiP)", "Search Reviews (WiP)"])

# -------------------------------
# TAB 1: APP RATINGS
# -------------------------------
with app_tab:
    st.subheader("Check evolution of Apps' rating over time")
    
    # Sidebar-like filter controls inside the tab
    with st.container():
        cols = st.columns([2, 3, 2])

        # 1) Bank App filter
        app_list = sorted(df["app"].dropna().unique().tolist())
        default_apps = app_list # show all by default
        selected_apps = cols[0].multiselect(
            "Bank App",
            options=app_list,
            default=default_apps,
        )


        # 2) Time Period slider
        min_date = pd.Timestamp('2016-01-01')
        max_date = pd.to_datetime(df["review_date"]).max()
        default_start = max(min_date, max_date - pd.DateOffset(years=5))        
        start_date, end_date = cols[1].slider(
            "Time Period",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
            format="YYYY-MM-DD",
        )


        # 3) Time Unit
        unit = cols[2].selectbox(
            "Time Unit",
            options=["Month", "Quarter", "Semester", "Year"],
            index=1,
        )

    # Apply filters
    mask = (
        df["review_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & (df["app"].isin(selected_apps) if selected_apps else True)
    )
    df_f = df.loc[mask].copy()

    if df_f.empty:
        st.info("No data for the selected filters.")
        st.stop()

    # Aggregate
    agg = aggregate_avg_rating(df_f, unit)

    # Small KPIs – move above chart
    k1, k2, k3 = st.columns(3)
    k1.metric("Reviews (filtered)", f"{len(df_f):,}")
    k2.metric("Apps selected", f"{len(selected_apps)}")
    k3.metric("Avg. rating", f"{df_f['score'].mean():.2f} / 5")

    # Chart (Altair)
    # - Build base palette per app (unordered)
    palette = build_brand_palette(sorted(df_f["app"].dropna().unique().tolist()))
    # - Compute latest avg_score per app in the aggregated data and order desc
    latest = agg.sort_values("period").groupby("app", as_index=False).tail(1)
    legend_order = latest.sort_values("avg_score", ascending=False)["app"].tolist()
    # - Align color range to that domain order so colors match apps
    color_range = palette_in_order(legend_order, palette)
    base = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X("period:T", title="Period"),
        y=alt.Y("avg_score:Q", title="Average rating"),
        color=alt.Color(
            "app:N",
            title="App",
            sort=legend_order,
            scale=alt.Scale(domain=legend_order, range=color_range),
            legend=alt.Legend(title="App"),
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