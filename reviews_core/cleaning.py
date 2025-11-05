import pandas as pd
import numpy as np
import re
from pathlib import Path


def clean_new_reviews():
    
    """
    Receives an input_df with raw data, cleans it into a consistent schema and returns the cleaned DataFrame.
    """
    
    # Load new_df_raw.parquet with new raw reviews for cleaning
    load_path = Path("assets/dfs_pipeline/new_df_raw.parquet")
    df = pd.read_parquet(load_path)

    # Print original df_shape
    print(f"Original DataFrame shape: {df.shape}")

    # Drop columns we don't keep
    df = df.drop(columns = ['app_id','reviewId','user_name','thumbs_up','Reply','Reply_Date','App_Version'])
    
    # Parse 'date' column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Rename columns
    colmap = {
        'app_name': 'app',
        'text': 'review_text',
        'date': 'review_date',
    }
    new_df_cleaned = df.rename(columns=colmap)

    # Print df_cleaned shape
    print(f"Cleaned DataFrame shape: {new_df_cleaned.shape}")

    # Save new_df_cleaned
    new_cleaned_path = Path("assets/dfs_pipeline/new_df_clean.parquet")
    new_df_cleaned.to_parquet(new_cleaned_path, index=False)
    print(f"✅ Saved cleaned new reviews → {new_cleaned_path}")

    return new_df_cleaned


def update_df_clean(): 
   
    """
    Updates the existing cleaned DataFrame with new cleaned reviews.
    Loads the existing cleaned DataFrame from a Parquet file, appends the new cleaned reviews,
    and saves the updated DataFrame back to the Parquet file in dfs_pipeline/df_clean.parquet
    """

    # Load existing cleaned DataFrame
    existing_df_clean_path: str = "assets/dfs_pipeline/df_clean.parquet"
    existing_df_clean = pd.read_parquet(existing_df_clean_path)

    # Load new cleaned reviews
    new_df_clean_path: str = "assets/dfs_pipeline/new_df_clean.parquet"
    new_df_clean = pd.read_parquet(new_df_clean_path)
    
    # Print shapes before concatenation
    print(f"Existing cleaned DataFrame shape: {existing_df_clean.shape}")
    print(f"New cleaned DataFrame shape: {new_df_clean.shape}")

    # Concatenate existing and new cleaned DataFrames
    updated_df_clean = pd.concat([existing_df_clean, new_df_clean], ignore_index=True)
    print(f"Updated cleaned DataFrame shape: {updated_df_clean.shape}")

    # Save updated cleaned DataFrame back to Parquet
    updated_df_clean.to_parquet(existing_df_clean_path, index=False)
    print(f"✅ Updated cleaned DataFrame saved to {existing_df_clean_path}")
    return updated_df_clean