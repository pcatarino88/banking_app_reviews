import pandas as pd
import numpy as np
from google_play_scraper import app, Sort, reviews_all
import time

def scrape_reviews():
    
    # List of APPs' packages to be scraped
    apps = {
        "Santander UK": "uk.co.santander.santanderUK",
        "HSBC": "uk.co.hsbc.hsbcukmobilebanking",
        "LLoyds": "com.grppl.android.shell.CMBlloydsTSB73",
        "Barclays": "com.barclays.android.barclaysmobilebanking",
        "Revolut": "com.revolut.revolut",
        "Monzo": "co.uk.getmondo"
    }

    # Loop through the apps and scrape reviews
    all_reviews = []

    for app_name, app_id in apps.items():
        start_time = time.time()
        try:
            reviews = reviews_all(
                app_id,
                sleep_milliseconds=100,  
                lang='en',
                country='gb'
            )
            for review in reviews:
                all_reviews.append({
                    "app_name": app_name,
                    "user_name": review["userName"],
                    "score": review["score"],
                    "text": review["content"],
                    "date": review["at"],
                    "thumbs_up": review["thumbsUpCount"],
                    "Reply":review['replyContent'],
                    'Reply_Date':review['repliedAt'],
                    'App_Version':review['appVersion']
                })
            
            elapsed = time.time() - start_time
            print(f"✅ Fetched reviews for {app_name} in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Error fetching reviews for {app_name}: {e}")
    
        time.sleep(2)

    # Save to CSV
    raw_reviews = pd.DataFrame(all_reviews)
    filename = "1_df_raw.csv"
    path = fr"C:\Users\pedro\OneDrive\Escritorio\Projetos\Banking APPs Reviews\{filename}"
    raw_reviews.to_csv(path, index=False)
    print(f"✅ Done! Reviews saved to {filename}")