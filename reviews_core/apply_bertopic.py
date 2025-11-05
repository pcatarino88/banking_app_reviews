import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


def bertopic_cleaning():
    """
    Loads new_df_clean.parquet
    Adds a column with 'word_count' to the DataFrame.
    """
    
    # Load new reviews
    new_df_clean = pd.read_parquet("../assets/dfs_pipeline/new_df_clean.parquet")
    
    # Add word count column
    new_df_clean['word_count'] = new_df_clean['review_text'].apply(lambda x: len(str(x).split()))

    # Drop reviews with less than 4 words
    print(f"Shape before dropping reviews with less than 4 words: {new_df_clean.shape}")
    new_df_clean = new_df_clean[new_df_clean['word_count'] >= 4].reset_index(drop=True)
    print(f"Shape after dropping reviews with less than 4 words: {new_df_clean.shape}")

    # Setting basic STOP_WORDS to favor topics related to functionality issues
    PRAISE_WORDS = [
    "amazing", "awesome", "brilliant", "excellent", "fantastic", "love", "nice", "perfect", "star", "stars",
    "awful", "bad", "disappointing", "horrible", "poor", "terrible", "useless", "worst", "best"
    ]
    CUSTOM_STOP_WORDS = [
    "santander", "revolut", "revolute", "revoult", "revlout", "revelout", "hsbc", "barclays", 
    "barclay", "lloyds","lloyd", "monzo", "app", "banking", "bank"
    ]
    STOP_WORDS = PRAISE_WORDS + CUSTOM_STOP_WORDS

    #def bertopic_cleaning(df: pd.DataFrame, col: str = "review_text", name="df"):

    print(f"Shape of new_df_clean before bertopic cleaning: {new_df_clean.shape}")

    new_df_clean["bert_review_text"] = new_df_clean['review_text'].astype(str) # Create a new column for cleaned text

    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"http\S+", "", str(x))) # Remove URLs
    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip()) # Normalize spaces

    for w in STOP_WORDS:
        new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(rf"\b{re.escape(w)}\b", "", x, flags=re.IGNORECASE)) # Remove stop words

    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip()) # Normalize spaces again after stop word removal

    # elimnate rows with less than 2 words after cleaning
    new_df_clean = new_df_clean[new_df_clean["bert_review_text"].str.split().str.len() >= 2].copy()

    # print number of rows after cleaning
    print(f"Shape of new_df_clean after bertopic cleaning: {new_df_clean.shape}")

    # save new_df_topics
    new_df_topics = new_df_clean.copy()
    new_df_topics.to_parquet("../assets/dfs_pipeline/new_df_topics.parquet", index=False)

    return new_df_topics


def apply_bertopic():
    """
    Loads the cleaned DataFrame and applies the trained BERTopic model to assign topics to reviews.
    Returns the DataFrame with topics.
    """
    # load cleaned DataFrame with bertopic cleaning applied
    new_df_topics = pd.read_parquet("../assets/dfs_pipeline/new_df_topics.parquet")

    # load bertopic model
    model_path = "../assets/models/bertopic/seed_final_model"
    bertopic_model = BERTopic.load(model_path, embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))

    # Apply bertopic model to get topics
    new_reviews = new_df_topics["bert_review_text"].tolist()
    new_topics, new_probs = bertopic_model.transform(new_reviews)
    new_df_topics["bert_topic"] = new_topics
    new_df_topics["bert_probs"] = new_probs.max(axis=1)
    print(f"Applied BERTopic model. Shape of new_df_topics: {new_df_topics.shape}")

    # Apply custom_labels
    custom_labels = {
        -1: "Outliers",
        0: "Undefined",
        1: "Simplicity",
        2: "Money Management",
        3: "Usability & Experience",
        4: "Security & Close Account",
        5: "Login & Authentication",    
        6: "Travel & FX",
        7: "Reliability",
        8: "Cards",    
        9: "Customer Service",
        10: "Compatibility & Launch Issues",
        11: "Stability",
        12: "Layout & Interface",
        13: "Cheque",
        14: "Investments & Fees",
        15: "Updates",
        16: "Functional Bugs",
        17: "Notifications & Ads",
        18: "Chat",
        19: "Referral Program"
    } 
    
    new_df_topics['bert_label'] = new_df_topics['bert_topic'].map(custom_labels)

    # Remove reviews labeled as 'Outliers' or 'Undefined'
    print(f"Shape before removing 'Outliers' and 'Undefined': {new_df_topics.shape}")
    new_df_topics = new_df_topics[~new_df_topics['bert_label'].isin(['Outliers', 'Undefined'])].reset_index(drop=True)
    print(f"Shape after removing 'Outliers' and 'Undefined': {new_df_topics.shape}")

    # Apply macro labels
    macro_labels = {
        "Simplicity": "User Experience",
        "Usability & Experience": "User Experience",
        "Layout & Interface": "User Experience",
        "Notifications & Ads": "User Experience",
        "Cards": "Products",
        "Investments & Fees": "Products",
        "Cheque": "Products",
        "Referral Program": "Products",
        "Customer Service": "Customer Service",
        "Chat": "Customer Service",
        "Reliability": "Performance",
        "Stability": "Performance",
        "Compatibility & Launch Issues": "Performance",
        "Updates": "Performance",
        "Functional Bugs": "Performance",
    }

    new_df_topics['bert_macro_label'] = new_df_topics['bert_label'].map(macro_labels).fillna(new_df_topics['bert_label'])

    # Apply 'subset' value as 'new reviews'
    new_df_topics['subset'] = 'new reviews'

    # Set columns and order for final new_df_topics
    cols = ['subset','app','score','review_text','review_date','word_count','bert_macro_label','bert_label','bert_probs','scrape_date']
    new_df_topics = new_df_topics[cols]

    # Save new_df_topics as excel (remove timezone from review_date)
    for col in new_df_topics.select_dtypes(include=["datetimetz"]).columns:
        new_df_topics[col] = new_df_topics[col].dt.tz_localize(None)
    new_df_topics.to_excel("../assets/dfs_pipeline/new_df_topics.xlsx", index=False)

    # Save new_df_topics as parquet
    new_df_topics.to_parquet("../assets/dfs_pipeline/new_df_topics.parquet", index=False)

    return new_df_topics