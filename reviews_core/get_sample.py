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
        
    # --- Prioritized sampling ---------------------------------------------------
    high = filtered[filtered['topic_prob_SEG'] >= 0.5]
    low  = filtered[filtered['topic_prob_SEG'] < 0.5]
    
    rng = np.random.default_rng(seed)
    n_high = min(len(high), n)
    n_low = max(0, n - n_high)
    
    # createa a int random_state from the Generator
    rs_high = int(rng.integers(0, 1_000_000_000)) if len(high) else None
    rs_low  = int(rng.integers(0, 1_000_000_000)) if len(low)  else None

    n_high = min(len(high), n)
    n_low  = max(0, n - n_high)
    
    take_high = high.sample(n=n_high, random_state=rs_high) if n_high > 0 else high.iloc[[]]
    take_low  = low.sample(n=n_low,  random_state=rs_low)  if n_low  > 0 else low.iloc[[]]

    out = pd.concat([take_high, take_low], ignore_index=True)

    # --- Return only requested columns (if you want) ---
    cols = ['app', 'review_text', 'score', 'review_date', 'topic_label_SEG']
    cols = [c for c in cols if c in out.columns]
    return out[cols]