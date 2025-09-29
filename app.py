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
from openai import OpenAI


# -------------------------------
# I. Page config
# -------------------------------

st.set_page_config(
    page_title="Banking App Reviews",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="auto"
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
        /* Target sidebar headers */
    [data-testid="stSidebar"] {
        background-color: #306F82;   /* Blue Green */
        color: white !important;        
    }
    """,
    unsafe_allow_html=True,
)

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


NEGATIVE_TOPICS = ["Login & Security","Payments & Cards","Updates & Crashes",
                   "Transfers & Transact","Interface & Fees","Customer Support"]    


POSITIVE_TOPICS = ["Usability","Features & Security","Transactions & Money Mgmt",
                   "Investments","Cards & Travel","Updates & Support"]



# -------------------------------
# MAIN
# -------------------------------

st.title("📱 App Reviews")
c1,c2 = st.columns([6,2])
st.caption("Interactive analysis of App store reviews of UK banks")

# Tabs
app_tab, topics_tab, reviews_tab = st.tabs(["App Ratings", "Key Topics", "Search Reviews"])

# ===============================
# TAB 1: APP RATINGS 
# ===============================
with app_tab:

    # LOAD DF_TAB1
    df_tab1= load_df("assets/df_tab1.parquet")        

    # ----------------------------------
    # FILTERS 
    # ----------------------------------
    c1, c2, c3, c4, c5 = st.columns([2, 0.1, 2, 0.1, 1])

    # Bank Filter
    with c1:
        app_list = sorted(df_tab1["app"].dropna().unique().tolist())
        selected_apps = st.multiselect(
            "Bank App", options=app_list, default=app_list,
            help="Choose one or more apps/banks."
        )

    # Time Period slider
    with c3: 
        min_date = df_tab1["period_month"].min()
        max_date = df_tab1["period_month"].max()
        default_start = max(min_date, max_date - pd.DateOffset(years=4))
        start_date, end_date = st.slider(
            "Time Period",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
            format="YYYY-MM-DD",
            help="Filter by review date."
        )

    # Time Unit 
    with c5:
        unit = st.selectbox(
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
            legend=alt.Legend(
                title="App",
                orient='right',
                #direction = "horizontal",
                labelFontSize=12,
                titleFontSize=13,
                padding=7, # gap between legend and chart
                symbolSize=80),
        ),
        tooltip=[
            alt.Tooltip("period:T", title="Period"),
            alt.Tooltip("app:N", title="App"),
            alt.Tooltip("avg_score:Q", title="Avg. rating", format=".2f"),
        ],
    ).properties(height=420)

    st.write("")

    st.altair_chart(base, use_container_width=True)

    # ------------------
    # Tab 1 Footer
    # ------------------
    st.write("")
    st.markdown(
        """
    <div style="text-align:left; color: gray; font-size: 10px; margin-left:10px; margin-top:5px;">
        Note: Ratings for each time unit are simple averages of the monthly averages - i.e., not weighted by review counts.
        </a><br>
    </div>
        """,
        unsafe_allow_html=True
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
    c1, s1, c2, s2, c3 = st.columns([1, 0.1, 2, 0.1, 2])

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
    with c2:
        apps = sorted(df_tab2["app"].dropna().unique().tolist())
        sel_apps = st.multiselect(
            "Bank App",
            options=apps,
            default=apps,
            help="Choose one or more apps/banks.",
            key="t2_bank_app"
        )
    
    # --- Time Period slider
    with c3:        
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
        margin=dict(l=5, r=5, t=30, b=60),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------
    # Tab 2 Footer
    # ------------------
    st.write("")
    st.markdown(
        """
    <div style="text-align:left; color: gray; font-size: 10px; margin-left:10px; margin-top:5px;">
        Notes: Proportions are within each app (stacked to 100%). ‘Positive’ uses scores 4–5 and ‘Negative’ uses scores 1–2. 
        Reviews that were not possible to allocate to a specific topic were removed from this analysis.
        </a><br>
    </div>
        """,
        unsafe_allow_html=True
    )  


# -------------------------------
# Tab 3 SEARCH REVIEWS
# -------------------------------

with reviews_tab:

    # --- LOAD DF_TAB3 -------------------------

    df_tab3 = load_df("assets/df_tab3.parquet")
    
    # --- Filters row ---------------------------
    c1, s1, c2, s2, c3 = st.columns([1, 0.1, 2, 0.1, 2])

    # Type of review - default 'Negative'
    with c1:
        sentiment_t3 = st.segmented_control(
            "Type of reviews",
            ["Negative", "Positive"],
            default = "Negative",
            key="t3_sentiment"
        )
    
    # Bank App select (optional)
    with c2:
        app_options = ["All"] + sorted([a for a in df_tab3["app"].dropna().astype(str).unique()])
        app_sel = st.selectbox("Bank App (optional)", app_options, index=0)
    
    # Topic select (optional)
    with c3:
        if sentiment_t3 == "Negative":
            topic_options = ["All"] + NEGATIVE_TOPICS
        else:
            topic_options = ["All"] + POSITIVE_TOPICS

        topic_sel = st.selectbox("Topic (optional)", topic_options, index=0)

    st.write("")

    # --- Search row ---------------------------
    c4, s3, c5 = st.columns([4, 0.1, 1.5])

    with c4:
        words_raw = st.text_input(
            "Words to search (comma or semicolon separated)",
            placeholder="e.g., fees, login, customer service"
        )
    with c5:
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

        # Words filter with OR logic - supports multiple words to search with OR logic
        words = [w.strip() for w in re.split(r"[,\n;]+", words_raw) if w.strip()]
        if words:
            if isinstance(words, str):
                words = [words]  
            
            pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
            mask = df_filtered['review_text'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            df_filtered = df_filtered[mask]

        # Safety: if nothing left, message and stop
        if df_filtered.empty:
            st.info("No reviews found for the selected filters / words.")
            st.stop()

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
            st.stop()

        if out.empty:
            st.info("No reviews returned by the sampler with the current settings.")
            st.stop()
            
        st.caption(f"Showing up to {n_reviews} reviews.")
        for i, r in out.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.markdown(f"**App:** {r['app']}")
                c2.markdown(f"**Score:** {r['score']}")
                c3.markdown(f"**Date:** {r['review_date']}")
                st.markdown(f"**Topic:** {r['topic_label_SEG']}")
                st.markdown(r["review_text"])  # full text, wrapped
            #st.write("")  # small spacer)

        # Export to csv option
        st.download_button(
            "Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="review_samples.csv",
            mime="text/csv",
        )

# ================================================
# CHAT LLM (SIDEBAR)
# ================================================

OPENAI_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------------------------
# LLM helpers
# -------------------------------------------------

# App aliases
BANK_ALIASES = {
    "HSBC":     ["hsbc", "hsbc uk", "hsbc bank"],
    "Santander":["santander", "santander uk"],
    "Barclays": ["barclays", "barclays uk"],
    "Lloyds":   ["lloyds", "lloyds bank", "loyds"],
    "Monzo":    ["monzo"],
    "Revolut":  ["revolut"]
}

def _ensure_llm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has the columns the LLM helpers expect."""
    out = df.copy()

    # row_id
    if "row_id" not in out.columns:
        out = out.reset_index(drop=True)
        out["row_id"] = np.arange(len(out), dtype=int)

    # review_date (datetime -> date string)
    if "review_date" in out.columns:
        out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce").dt.date
    else:
        out["review_date"] = pd.NaT

    # app / score / review_text safe defaults
    for col, default in [("app", "unknown-app"),
                         ("score", np.nan),
                         ("review_text", "")]:
        if col not in out.columns:
            out[col] = default

    return out

def _detect_target_apps(question: str) -> set:
    q = (question or "").lower()
    hits = set()
    for app, aliases in BANK_ALIASES.items():
        if any(a in q for a in aliases):
            hits.add(app)
    return hits


def _pick_context_rows(df: pd.DataFrame, question: str, k: int = 12) -> pd.DataFrame:
    df = _ensure_llm_columns(df)  
    if df.empty or not isinstance(question, str) or not question.strip():
        return df.head(0)

    # 1) detect target apps from the question
    targets = _detect_target_apps(question)
    targets_lower = {t.lower() for t in targets}

    # 2) base masks
    text = df["review_text"].astype(str)
    words = [w.strip() for w in re.split(r"[\s,;:.!?/\\-]+", question) if w.strip()]
    text_mask = pd.Series(False, index=df.index)
    for w in words:
        m = text.str.contains(rf"\b{re.escape(w)}\b", case=False, na=False, regex=True)
        text_mask |= m

    # 3) app mask (case-insensitive exact match on canonical app names)
    app_series = df["app"].astype(str)
    app_mask = app_series.str.lower().isin(targets_lower) if targets else pd.Series(False, index=df.index)

    # 4) combine logic
    if targets:
        # if the user asked about a specific app, prefer that app; include rows even if text doesn't match
        mask = (app_mask & text_mask) | app_mask
    else:
        mask = text_mask

    hits = df[mask].copy()
    if hits.empty and targets:
        # still return that app's reviews if keywords missed
        hits = df[app_mask].copy()
    if hits.empty:
        hits = df.copy()

    hits = hits.sort_values("review_date", ascending=False).head(k)
    cols = ["row_id", "app", "review_date", "score", "review_text"]
    return hits[cols]

def _rows_to_bullets(rows: pd.DataFrame, max_rows: int = 12, max_text: int = 260) -> str:
    rows = _ensure_llm_columns(rows).head(max_rows)
    if rows.empty:
        return "(no matching context)"

    out = []
    for _, r in rows.iterrows():
        rid = int(r["row_id"]) if pd.notna(r["row_id"]) else -1
        date_str = str(r["review_date"])[:10] if pd.notna(r["review_date"]) else "NA"
        app = str(r["app"])
        score = r["score"]
        txt = str(r["review_text"])[:max_text]
        out.append(f"- [row_id={rid}] {date_str} | {app} | score={score} :: {txt}")
    return "\n".join(out)

def ask_llm_openai(question: str, context_bullets: str):
    """OpenAI call (uses Streamlit secrets, falls back to env var)."""
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    client = OpenAI(api_key=api_key)

    system = (
        "You are an assistant named PAI that stands for Pedro Artificial Intelligence. "
        "If asked about your identity, respond that you are PAI, which stands for Pedro Artificial Intelligence, and that you are an AI assistant created by Pedro Catarino. "
        "Your role is to analyze banking app store reviews. "
        "The source of the reviews is Google Play Store and is limited to UK banks."
        "Be concise, numeric when helpful, and call out uncertainty if data is thin."
        "Provide example of citations from reviews when relevant."
        "Use only the provided context bullets to ground your answer. "
        "If the user asks a question unrelated to these reviews "
        "(for example, general knowledge, or anything not grounded in the context), "
        "politely refuse and remind them that you can only discuss insights from the reviews."
        "As for now, you are not prepared to answer to questions that require information about dates/periods of analysis,"
        "If asked about dates/periods, please respond with 'I am still not prepared to give you details about dates, but I expect to be able to do it soon!! 😊'"
    )
    user = f"""Question: {question}

Context:
{context_bullets}
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=500,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )

    answer = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    meta = None
    if usage:
        prompt_t = usage.prompt_tokens or 0
        comp_t = usage.completion_tokens or 0
        cost = (prompt_t * 0.60 / 1_000_000) + (comp_t * 2.40 / 1_000_000)  # gpt-4o-mini
        meta = {"prompt_tokens": prompt_t, "completion_tokens": comp_t, "cost_usd": round(cost, 6)}
    return answer, meta

# -------------------------------------------------
# Chat UI
# -------------------------------------------------

def sidebar_chat_single_turn(df_tab3: pd.DataFrame, key: str = "sidebar-single"):
    
    st.sidebar.header("😊 Ask PAI")

    # ---- UI state (single-turn) ----
    ss = st.session_state
    ss.setdefault(f"{key}_phase", "idle")        # idle | thinking | answered
    ss.setdefault(f"{key}_q", "")                # question text
    ss.setdefault(f"{key}_answer", "")           # answer text
    ss.setdefault(f"{key}_ctx", "")              # context bullets

    # small CSS tweak for compact look
    st.markdown("""
    <style>
      /* tighten the card around the input area */
      .ask-wrap { 
        border-radius: 5px;
        padding: 8px 10x 10px;   /* smaller padding = less space to the border */
        margin-bottom: 10px;
      }
      .hint { font-size: 0.85rem; opacity: 0.85; margin: 0 0 6px 0; }
      /* make the input taller and comfy */
      [data-testid="stTextArea"] textarea {
        min-height: 120px;    /* adjust height here */
        line-height: 1.35;
        padding: 10px 12px;   /* inner padding of the black box */
        color: white;         /* text color */            
      }
    /* Text color inside the textarea */
    [data-testid="stTextArea"] textarea {
        color: #1A1818 !important;   /* typed text */
        background: #ffffff !important;  /* box background */
        font-size: 1rem !important;  
    }
    /* Placeholder text color */
    [data-testid="stTextArea"] textarea::placeholder {
        color: #999999 !important;   /* placeholder */                
    .ans { font-size: 0.98rem; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

    # --- Composer (textarea so we can set height & avoid browser autocomplete) ---
    st.markdown("<div class='hint'>Ask about ratings, topics, or anything from the reviews…</div>", unsafe_allow_html=True)
    with st.form(key=f"{key}_form", clear_on_submit=False):
        st.markdown("<div class='ask-wrap'>", unsafe_allow_html=True)
        ss[f"{key}_q"] = st.text_area(
            label="",
            value=ss[f"{key}_q"],
            height=120,  # <- increase/decrease as you like
            key=f"{key}_textarea_v2",  # new key helps avoid old browser suggestions
            placeholder="Type your question…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("➤", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and ss[f"{key}_q"].strip() and ss[f"{key}_phase"] in ("idle", "answered"):
        ss[f"{key}_phase"] = "thinking"
        ss[f"{key}_answer"] = ""
        ss[f"{key}_ctx"] = ""
        st.rerun()

    # Thinking → Answer (keep your existing logic below)
    if ss[f"{key}_phase"] == "thinking":
        ph = st.empty()
        ph.markdown("<p style='color:white; font-weight:bold;'>Thinking...</p>",unsafe_allow_html=True)
        ctx_df = _pick_context_rows(df_tab3, ss[f"{key}_q"], k=12)
        bullets = _rows_to_bullets(ctx_df, max_rows=12)
        try:
            answer, meta = ask_llm_openai(ss[f"{key}_q"], bullets)
        except Exception as e:
            answer, meta = f"LLM error: {e}\n(Do you have your API key set?)", None
        ss[f"{key}_answer"] = answer
        ss[f"{key}_ctx"] = bullets
        ss[f"{key}_phase"] = "answered"
        st.rerun()

    if ss[f"{key}_phase"] == "answered" and ss[f"{key}_answer"]:
        st.markdown(f"<div class='ans'>{ss[f'{key}_answer']}</div>", unsafe_allow_html=True)
        if st.button("Make a new question", use_container_width=True, key=f"{key}_reset", type="primary"):
            ss[f"{key}_phase"] = "idle"
            ss[f"{key}_q"] = ""
            ss[f"{key}_answer"] = ""
            ss[f"{key}_ctx"] = ""
            st.rerun()

        return  # stop rendering further on this run

    if ss[f"{key}_phase"] == "answered":
        # Show the final answer under the input
        st.markdown(f"<div class='ans'>{ss[f'{key}_answer']}</div>", unsafe_allow_html=True)

        # Reset button at the bottom
        if st.button("Make a new question", use_container_width=True, key=f"{key}_reset", type="primary"):
            ss[f"{key}_phase"] = "idle"
            ss[f"{key}_q"] = ""
            ss[f"{key}_answer"] = ""
            ss[f"{key}_ctx"] = ""
            st.rerun()


# --- render the sidebar (always available) ---
with st.sidebar:
    if "df_tab3" in locals() and isinstance(df_tab3, pd.DataFrame) and not df_tab3.empty:
        sidebar_chat_single_turn(df_tab3)
    else:
        st.header("🤓 Ask PAI")
        st.info("Load df_tab3 to enable chat.")

# ================================================
# APP FOOTER
# ================================================

# Add spacer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="margin: 5px 0 0 0; border: none; border-top: 1px solid #ddd;">
    """,
    unsafe_allow_html=True
)

# Footer note
st.markdown(
    """
<div style="text-align:left; color: gray; font-size: 12px; margin-left:10px; margin-top:0px;">
    Developed by 
    <a href="https://www.linkedin.com/in/pedrofcatarino/" target="_blank"
    style="color:#0a66c2; text-decoration:underline;">
    Pedro Catarino
    </a><br>
    Data source: Google Play Store reviews. Last Update: 4th September 2025.
</div>
    """,
    unsafe_allow_html=True
)

