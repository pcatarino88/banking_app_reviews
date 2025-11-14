import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


CUSTOM_WORDS = {'app','santander','revolut','hsbc','lloyds',
                'barclays','monzo','etc','much','bank', 'banking'}

stop_words = STOPWORDS.union(CUSTOM_WORDS)


def generate_wordcloud(df_filtered, stop_words, width=1000, height=300, 
                       colormap='viridis', background_color='white'):
    
    """
    Generate and display a word cloud from filtered dataframe using only
    bi- and tri-grams (2- and 3-word phrases). 
    Only considers reviews from the last 2 years.
    """

    # Ensure datetime
    df_filtered['review_date'] = pd.to_datetime(df_filtered['review_date'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['review_date'])
    if df_filtered.empty:
        st.warning("No valid review dates available.")
        return None
    
    # --- Filter to last 2 years based on most recent review date ---
    max_date = df_filtered['review_date'].max()
    cutoff_date = max_date - pd.DateOffset(years=2)
    df_recent = df_filtered[df_filtered['review_date'] >= cutoff_date]

    # Check if dataframe is empty
    if df_recent.empty:
        st.warning("No data available for word cloud generation.")
        return None
    
    # Check if review_text column exists
    if 'review_text' not in df_recent.columns:
        st.error("Column 'review_text' not found in dataframe.")
        return None
    
    # Combine all text from review_text column
    texts = df_recent['review_text'].dropna().astype(str).tolist()
    if not any(t.strip() for t in texts):
        st.warning("No text data available in 'review_text' column.")
        return None
    
    # --- Create bigram/trigram frequencies ---
    vect = CountVectorizer(
        ngram_range=(2, 3),        # only 2- and 3-word phrases
        stop_words=list(stop_words),
        token_pattern=r"(?u)\b[\w']+\b",
        lowercase=True,
        min_df=2                   # ignore very rare n-grams
    )
    X = vect.fit_transform(texts)
    if X.shape[1] == 0:
        st.warning("No bi/tri-grams found with current settings.")
        return None

    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = vect.get_feature_names_out()
    freq_dict = {vocab[i]: int(freqs[i]) for i in range(len(vocab))}
    
    # Generate word cloud from bigram/trigram frequencies
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=set(),        # already removed in CountVectorizer
        collocations=False,     # we are already using n-grams from countvectorizer 
        random_state=42
    ).generate_from_frequencies(freq_dict)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    return fig