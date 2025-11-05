import pandas as pd


def update_df_monthly():

    """
    Create/Update df_monthly DataFrame aggregating reviews by month and app.
    """

    # Load cleaned DataFrame
    clean_df_path: str = "assets/dfs_pipeline/df_clean.parquet"
    df_clean = pd.read_parquet(clean_df_path)

    # Convert 'review_date' to datetime if not already
    df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')

    # Extract year-month for aggregation
    df_clean['period_month'] = df_clean['review_date'].dt.to_period('M').dt.to_timestamp('M')
   
    # Aggregate reviews by month and app
    selected = df_clean[['app','period_month','score']].copy()
    selected['app'] = selected['app'].astype('category')

    df_monthly = (selected
        .groupby(['period_month','app'], observed=True)['score']
        .agg(avg_score='mean', n_reviews='size')
        .reset_index()
    )

    # Save df_monthly to Parquet
    monthly_df_path: str = "assets/df_monthly.parquet"
    df_monthly.to_parquet(monthly_df_path, index=False)
    print(f"✅ Saved monthly aggregated DataFrame → {monthly_df_path}")

    return df_monthly


def update_df_topic():
    
    """
    Updates the existing DataFrame with topics by appending new reviews with topics.
    - Loads the existing Dataframe from "assets/df_topic.parquet"
    - Appends new reviews from "assets/dfs_pipeline/new_df_topics.parquet"
    - Saves the updated DataFrame back to "assets/df_topic.parquet"
    """

    # load existing DataFrame with topics
    existing_df_topic_path: str = "assets/df_topic.parquet"
    existing_df_topic = pd.read_parquet(existing_df_topic_path)

    # load new reviews with topics
    new_df_topics_path: str = "assets/dfs_pipeline/new_df_topics.parquet"
    new_df_topics = pd.read_parquet(new_df_topics_path)

    # print shapes before concatenation
    print(f"Existing df_topic shape: {existing_df_topic.shape}")
    print(f"New df_topics shape: {new_df_topics.shape}")

    # concatenate DataFrames
    updated_df_topic = pd.concat([existing_df_topic, new_df_topics], ignore_index=True)

    # save updated DataFrame
    updated_df_topic.to_parquet("assets/df_topic.parquet")

    print(f"✅ Updated df_topic saved to assets/df_topic.parquet. New shape: {updated_df_topic.shape}")

    return updated_df_topic