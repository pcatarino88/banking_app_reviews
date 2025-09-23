from __future__ import annotations
from datetime import datetime
import io
import os
import requests
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
from reviews_core import get_sample
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
    /* Adjust st.mnultiselect items size and font*/
    .stMultiSelect [data-baseweb="tag"],
    [data-baseweb="tag"] {
        font-size: 12px !important;     /* ↓ make text smaller */
        line-height: 1.1 !important;
        padding: 2px 6px !important;     /* tighter chip */
    }
    .stMultiSelect [data-baseweb="tag"] span {
        font-size: 12px !important;      /* make inner span smaller too */
    }
    /* The “x” icon size */
    .stMultiSelect [data-baseweb="tag"] svg {
        width: 12px !important;
        height: 12px !important;
    }
    
    /* dropdown option size */
    .stMultiSelect [data-baseweb="list"] div[role="option"] {
        font-size: 13px !important;
    }
    """,
    unsafe_allow_html=True
)

def show_memory_usage():
    p = psutil.Process(os.getpid())
    st.sidebar.caption(f"RAM used: {p.memory_info().rss / 1024**2:.0f} MB")

# -------------------------------
# II. Helpers
# -------------------------------

@st.cache_data 
def load_df(path: str, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=cols)

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

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


# -------------------------------
# MAIN
# -------------------------------
def main():
    show_memory_usage()

    st.title("📱 App Reviews")
    st.caption("Interactive analysis of App store reviews of UK banks")

    # Tabs
    app_tab, topics_tab, reviews_tab = st.tabs(
        ["App Ratings", "Key Topics", "Search Reviews"]
    )
    
    # -------------------------------
    # TAB 1: APP RATINGS 
    # -------------------------------
    with app_tab:
        
        # LOAD DF_TAB1
        df_tab1= load_df("assets/df_tab1.parquet")        

        with st.container():
            cols = st.columns([2, 0.1, 2, 0.1, 2])

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
            start_date, end_date = cols[2].slider(
                "Time Period",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD",
                help="Filter by review date."
            )

            # 3) Time Unit 
            unit = cols[4].selectbox(
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

        # --- BLANK SPACING
        st.write("")     

        # KPIs (use review counts from agg and weighted mean for rating)
        total_reviews = int(agg["n_reviews"].sum())
        weighted_avg = (
            (agg["avg_score"] * agg["n_reviews"]).sum() / max(total_reviews, 1)
        )

        k1, k2, k3, k4 = st.columns(4)
        k2.metric("Reviews (filtered)", f"{total_reviews:,}")
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
                    labelAngle=0,          # keep labels horizontal
                    labelOverlap=True
                ),
            ),
            y=alt.Y("avg_score:Q", title="Average rating"),
            color=alt.Color(
                "app:N",
                title="App",
                sort=legend_order,
                scale=alt.Scale(domain=legend_order, range=color_range),
                legend=alt.Legend(title="App",labelFontSize=12,titleFontSize=13,symbolSize=80,orient='top'),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="Period"),
                alt.Tooltip("app:N", title="App"),
                alt.Tooltip("avg_score:Q", title="Avg. rating", format=".2f"),
            ],
        ).properties(height=420)

        st.write("")

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

        # LOAD DF_TAB2
        
        cols_tab2 = ["app", "review_date", "score", "topic_label_SEG"]
        df_tab2= load_df("assets/df_tab3.parquet", cols=cols_tab2)
        
        # --- Light, memory-friendly dtypes
        df_tab2["review_date"]  = pd.to_datetime(df_tab2["review_date"], errors="coerce")
        df_tab2["score"] = pd.to_numeric(df_tab2["score"], errors="coerce").astype("Int64")

        # --- Categories reduce memory & speed up groupby
        for c in ("app", "topic_label_SEG"):
            df_tab2[c] = df_tab2[c].astype("category")
            
        
        # FILTERS 
        c1, c2, c3, c4, c5 = st.columns([1, 0.1, 2, 0.1, 2])

        # --- Type of review filter | default 'Negative'
        with c1:
            sentiment_t2 = st.segmented_control(
                "Type of reviews",
                ["Negative", "Positive"],
                default = "Negative",
                key="t2_sentiment"       
            )
            score_vals = [1, 2] if sentiment_t2 == "Negative" else [4, 5]
            
        # --- Bank App Filter
        with c3:
            apps = sorted(df_tab2["app"].dropna().unique().tolist())
            sel_apps = st.multiselect(
                "Bank App",
                options=apps,
                default=apps,
                help="Choose one or more apps/banks.",
                key="t2_bank_app"
            )
        
        # --- Time Period slider
        with c5:        
            min_dt = pd.to_datetime(df_tab2["review_date"].min())
            max_dt = pd.to_datetime(df_tab2["review_date"].max())
            default_start = max(min_dt, max_dt - pd.DateOffset(years=4))
            start_dt, end_dt = st.slider(
                "Time Period",
                min_value=min_dt.to_pydatetime(),
                max_value=max_dt.to_pydatetime(),
                value=(default_start.to_pydatetime(), max_dt.to_pydatetime()),
                format="YYYY-MM-DD",
                help="Filter by review date.",
                key="t2_time_period"
            )
        
    
        # --- Lightweight filtering ------------------------------
        mask = (
            df_tab2["app"].isin(sel_apps)
            & df_tab2["score"].isin(score_vals)
            & df_tab2["review_date"].between(pd.Timestamp(start_dt), pd.Timestamp(end_dt))
            & df_tab2["topic_label_SEG"].notna()
            & (df_tab2["topic_label_SEG"] != "Undefined")
        )
        
        view = df_tab2.loc[mask, ["app", "topic_label_SEG"]]
        if view.empty:
            st.info("No reviews match the current filters.")
            st.stop()
    
        # --- Prepare topic order & colors -----------------------------------------
        # Keep a stable topic order: use our color dict order where available, then any extras
        topics_in_view = view["topic_label_SEG"].cat.remove_unused_categories().cat.categories.tolist()
        ordered_topics = topics_in_view

        # Build a deterministic color map for the topics we actually have
        color_map = {t: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, t in enumerate(ordered_topics)}
        
        # Aggregate to proportions per app
        ct = (view.groupby(["app", "topic_label_SEG"], observed=True)
                  .size()
                  .rename("n")
                  .reset_index())
    
        # Ensure every (app, topic) pair exists -> aligned stacks
        all_index = pd.MultiIndex.from_product([sel_apps, ordered_topics], names=["app", "topic_label_SEG"])
        ct = ct.set_index(["app", "topic_label_SEG"]).reindex(all_index, fill_value=0).reset_index()
        
        totals = ct.groupby("app", as_index=False)["n"].sum().rename(columns={"n":"total_n"})
        ct = ct.merge(totals, on="app", how="left")
        ct["pct"] = np.where(ct["total_n"]>0, ct["n"]/ct["total_n"]*100.0, 0.0)
        
        x_order = sel_apps
        order_map = {a:i for i,a in enumerate(x_order)}
    
        # --- Build figure (stacked 100%) ------------------------------------------
        fig = go.Figure()
        for topic in ordered_topics:
            df_t = ct[ct["topic_label_SEG"]==topic].sort_values("app", key=lambda s: s.map(order_map))
            fig.add_trace(
                go.Bar(
                    x=df_t["pct"],
                    y=df_t["app"],
                    orientation = 'h',
                    name=topic,
                    marker_color=color_map[topic],
                    text=(df_t["pct"].round().astype(int).astype(str) + "%").where(df_t["pct"]>=7, ""),
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=10, color="white"),
                    hovertemplate=(
                        topic + ": %{x:.1f}%<br>"
                        + "Count: %{customdata}"
                        + "<extra></extra>"
                    ),
                    customdata=df_t["n"],
                )
            )
        
        fig.update_layout(
            barmode="stack",
            xaxis=dict(title="Proportion of reviews", range=[0, 100], ticksuffix="%", showgrid=True),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.4,
                xanchor="center", x=0.5,
                traceorder="normal",
                bgcolor="rgba(255,255,255,0.15)",
                font_size= 10,
                itemsizing="constant",        # makes marker a fixed size
                itemwidth=30,                 # width reserved for marker + spacing
                tracegroupgap=0 
            ),
            margin=dict(l=10, r=10, t=30, b=60),
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

        # --- LOAD DF_TAB3 -------------------------

        df_tab3 = load_df("assets/df_tab3.parquet")
       
        # --- Filters row ---------------------------
        c1, c2, c3, c4, c5 = st.columns([1, 0.1, 2, 0.1, 2])

        # Type of review - default 'Negative'
        with c1:
            sentiment_t3 = st.segmented_control(
                "Type of reviews",
                ["Negative", "Positive"],
                default = "Negative",
                key="t3_sentiment"
            )
      
        # Bank App select (optional)
        with c3:
            app_options = ["All"] + sorted([a for a in df_tab3["app"].dropna().astype(str).unique()])
            app_sel = st.selectbox("Bank App (optional)", app_options, index=0)
        
        # Topic select (optional)
        with c5:
            topic_options = ["All"] + sorted([t for t in df_tab3["topic_label_SEG"].dropna().astype(str).unique()])
            topic_sel = st.selectbox("Topic (optional)", topic_options, index=0)

        st.write("")

        # --- Search row ---------------------------
        c6, c7, c8 = st.columns([4, 0.1, 1.5])

        with c6:
            words_raw = st.text_input(
                "Words to search (comma / semicolon / newline separated)",
                placeholder="e.g., fees, login, customer service"
            )
        with c8:
            n_reviews = st.number_input("Number of reviews", min_value=1, max_value=10, value=5, step=1, help="Select number reviews to display" )

        st.write("")

        # --- Action button ---------------------------------------------------------
        do_search = st.button("Search Reviews", type="primary")
            
        if do_search:
            df_filtered = df_tab3.copy()
    
            # Sentiment to score buckets
            if sentiment_t3 == "Negative":
                df_filtered = df_filtered[df_filtered["score"].isin([1, 2])]
            else:  # Positive
                df_filtered = df_filtered[df_filtered["score"].isin([4, 5])]
    
            # App filter
            if app_sel != "All":
                df_filtered = df_filtered[df_filtered["app"].astype(str) == str(app_sel)]
    
            # Topic filter
            if topic_sel != "All":
                df_filtered = df_filtered[df_filtered["topic_label_SEG"].astype(str) == str(topic_sel)]
    
            # Words filter (supports multiple terms)
            words = [w.strip() for w in re.split(r"[,\n;]+", words_raw) if w.strip()]
            if words:
                if isinstance(words, str):
                    words = [words]  
                
                for w in words:
                    pattern = r"\b" + re.escape(w) + r"\b"
                    mask = df_filtered['review_text'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                    df_filtered = df_filtered[mask]
    
            # Safety: if nothing left, message and stop
            if df_filtered.empty:
                st.info("No reviews found for the selected filters / words.")
                return
    
            # --- Call your sampler (filters already applied so pass None for function parameters) -----------------------------
            try:
                out = get_sample(
                    df=df_filtered,
                    app=None,
                    words=None,
                    score=None,
                    topics=None,
                    n=n_reviews,
                    seed=st.session_state.get("seed", None),
                )
            except Exception as e:
                st.error(f"Error while sampling reviews: {e}")
                return
    
            if out.empty:
                st.info("No reviews returned by the sampler with the current settings.")
                return
               
            st.caption(f"Showing up to {n_reviews} reviews.")
            for i, r in out.iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    c1.markdown(f"**App:** {r['app']}")
                    c2.markdown(f"**Score:** {r['score']}")
                    c3.markdown(f"**Date:** {r['review_date']}")
                    st.markdown(f"**Topic:** {r.get('topic_label_SEG', r.get('topic_label_SEL','—'))}")
                    st.markdown(r["review_text"])  # full text, wrapped
                #st.write("")  # small spacer)
    
            # Optional: quick export
            st.download_button(
                "Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="review_samples.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
