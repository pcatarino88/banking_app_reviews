# run_pipeline.py

import logging
from datetime import datetime

from reviews_core.scraper import scrape_new_reviews
from reviews_core.cleaning import clean_new_reviews, update_df_clean
from reviews_core.apply_bertopic import bertopic_cleaning, apply_bertopic
from reviews_core.update_final_dataframes import update_df_monthly, update_df_topic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline():
    logging.info("🚀 Starting reviews pipeline")

    # 1) SCRAPE NEW REVIEWS
    try:
        logging.info("1/7 Scraping new reviews...")
        scrape_new_reviews()
    except Exception as e:
        logging.error(f"Error in scrape_new_reviews: {e}")
        return  # no point continuing if we can’t scrape

    # 2) CLEAN NEW REVIEWS
    try:
        logging.info("2/7 Cleaning new reviews...")
        clean_new_reviews()
    except Exception as e:
        logging.error(f"Error in clean_new_reviews: {e}")

    # 3) UPDATE DF_CLEAN
    try:
        logging.info("3/7 Updating df_clean...")
        update_df_clean()
    except Exception as e:
        logging.error(f"Error in update_df_clean: {e}")

    # 4) PREP FOR BERTOPIC
    try:
        logging.info("4/7 Preparing data for BERTopic...")
        bertopic_cleaning()
    except Exception as e:
        logging.error(f"Error in bertopic_cleaning: {e}")

    # 5) APPLY BERTOPIC
    try:
        logging.info("5/7 Applying BERTopic...")
        apply_bertopic()
    except Exception as e:
        logging.error(f"Error in apply_bertopic: {e}")

    # 6) UPDATE MONTHLY DF
    try:
        logging.info("6/7 Updating monthly dataframe...")
        update_df_monthly()
    except Exception as e:
        logging.error(f"Error in update_df_monthly: {e}")

    # 7) UPDATE TOPIC DF
    try:
        logging.info("7/7 Updating topic dataframe...")
        update_df_topic()
    except Exception as e:
        logging.error(f"Error in update_df_topic: {e}")

    logging.info("✅ Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()
