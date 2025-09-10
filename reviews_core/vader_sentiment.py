import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def add_vader_sentiment(
    df: pd.DataFrame,
    text_col: str = "review_text",
    score_col: str = "review_sentiment",
    analyzer: SentimentIntensityAnalyzer | None = None,
    inplace: bool = True,
):
    """
    Adds two columns:
      - '*_sentiment': VADER compound score in [-1, 1] (NaN for empty/invalid)
      - '*_sentiment_label': 'positive' | 'negative' | 'neutral' | 'undefined'
      - Consistent (int): 1 if sentiment label aligns with numeric score, else 0
    If we want to analyse both review and reply sentiment we have to run the function twice adapting text_col and score_col parameters
    """
    if not inplace:
        df = df.copy()

    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    # 1) sentiment scores (numeric; NaN for undefined)
    def _score(text):
        if not isinstance(text, str) or text.strip() == "":
            return np.nan
        return analyzer.polarity_scores(text)["compound"]

    df[score_col] = [ _score(t) for t in df[text_col] ]

    # 2) sentiment labels
    def _label(s):
        if pd.isna(s):
            return "undefined"
        if s >= 0.10:
            return "positive"
        if s <= -0.10:
            return "negative"
        return "neutral"

    label_col = f"{score_col}_label"
    df[label_col] = pd.Categorical([ _label(s) for s in df[score_col] ],
                                   categories=["negative","neutral","positive","undefined"])
    
    # 3) Consistency check
    def _consistent(row):
        if row[label_col] == "positive" and row["score"] in (0, 1):
            return 0
        if row[label_col] == "negative" and row["score"] in (4, 5):
            return 0
        return 1

    df["Consistent"] = df.apply(_consistent, axis=1).astype(int)

    return df