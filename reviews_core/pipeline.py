from reviews_core.cleaning import cleaning

def run_cleaning():
    print("🔹 Loading raw data...")
    df_raw = pd.read_parquet("data/raw/df_raw.parquet")

    print("🔹 Cleaning data...")
    df_clean = cleaning(df_raw)

    print("💾 Saving cleaned data...")
    df_clean.to_parquet("data/processed/df_clean.parquet", index=False)

    print("✅ Cleaning step completed successfully.")