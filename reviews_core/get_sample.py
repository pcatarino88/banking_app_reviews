import pandas as pd
import numpy as np

def get_sample(
    df: pd.DataFrame,
    app: str | None = None,
    words: str | None = None,
    score: int | None = None,
    topics: str | None = None,
    n: int = 5,
    seed: int | None = None,
    ):

    filtered = df.copy()
    
    # --- Apply filters ----------------------------------------------------------
    if app:
        m = filtered['app'].astype(str).str.contains(app, case=False, na=False)
        filtered = filtered[m]

    if words:
        pattern = r"\b" + re.escape(words) + r"\b"
        m = filtered['review_text'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
        filtered = filtered[m]

    if score is not None:
        m = filtered['score'].astype('Int64') == int(score)
        filtered = filtered[m]

    if topics:
        m = filtered['topic_label_SEG'].astype(str).str.contains(topics, case=False, na=False)
        filtered = filtered[m]
        
    # --- Prioritized sampling ----------------------------------------------------
    prob = filtered['topic_prob_SEG'].astype(float)
    txtlen = filtered['review_text'].astype(str).str.len()

    # Priority tresholds
    high = filtered[(prob >= 0.50) & (txtlen >= 100)]
    mid  = filtered[(prob >= 0.50) & (txtlen >= 60) % (txtlen < 100)]

    mid  = mid.loc[~mid.index.isin(high.index)]
    low  = filtered.loc[~filtered.index.isin(high.index.union(mid.index))]

    # Sort each tier by most recent first
    high = high.sort_values(by='review_date', ascending=False)
    mid  = mid.sort_values(by='review_date',  ascending=False)
    low  = low.sort_values(by='review_date',  ascending=False)

    # Take deterministically (High → Mid → Low)
    remaining = n
    parts = []
    if remaining > 0 and not high.empty:
        take = high.head(remaining)
        parts.append(take)
        remaining -= len(take)

    if remaining > 0 and not mid.empty:
        take = mid.head(remaining)
        parts.append(take)
        remaining -= len(take)

    if remaining > 0 and not low.empty:
        take = low.head(remaining)
        parts.append(take)
        remaining -= len(take)

    out = pd.concat(parts, ignore_index=True) if parts else filtered.iloc[0:0]

    # Return only requested columns
    cols = ['app', 'review_text', 'score', 'review_date', 'topic_label_SEG', 'topic_prob_SEG']
    cols = [c for c in cols if c in out.columns]
    return out[cols]