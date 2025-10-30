import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


CUSTOM_WORDS = {'app','santander','revolut','hsbc','lloyds',
                'barclays','monzo','etc','much','bank', 'banking',
                'use','still','want','need'}

stop_words = STOPWORDS.union(CUSTOM_WORDS)


def generate_wordcloud(df_filtered, stop_words, width=1000, height=300, 
                       colormap='viridis', background_color='white'):
    """
    Generate and display a word cloud from filtered dataframe.
    
    Args:
        df_filtered: Filtered pandas DataFrame
        stop_words: Set of stop words to exclude
        width, height: Dimensions of the word cloud
        colormap: Matplotlib colormap name
        background_color: Background color
    
    Returns:
        matplotlib figure object
    """
    # Check if dataframe is empty
    if df_filtered.empty:
        st.warning("No data available for word cloud generation.")
        return None
    
    # Check if review_text column exists
    if 'review_text' not in df_filtered.columns:
        st.error("Column 'review_text' not found in dataframe.")
        return None
    
    # Combine all text from review_text column
    text = ' '.join(df_filtered['review_text'].dropna().astype(str))
    
    # Check if text is empty after combining
    if not text.strip():
        st.warning("No text data available in 'review_text' column.")
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=stop_words,
        colormap=colormap,
        collocations=False,  # Avoid repeated phrases
        random_state=42
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    return fig