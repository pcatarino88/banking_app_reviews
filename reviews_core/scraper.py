import pandas as pd
from google_play_scraper import  reviews, Sort
from pathlib import Path
import time
from datetime import timezone

def scrape_reviews():
    
    # List of APPs' packages to be scraped
    apps = {
        "Santander": "uk.co.santander.santanderUK",
        "HSBC": "uk.co.hsbc.hsbcukmobilebanking",
        "LLoyds": "com.grppl.android.shell.CMBlloydsTSB73",
        "Barclays": "com.barclays.android.barclaysmobilebanking",
        "Revolut": "com.revolut.revolut",
        "Monzo": "co.uk.getmondo"
    }

    save_path = Path("../assets/intermediate_dfs/df_raw.parquet")
  
    # ---- Load existing data to know what we've already scraped ----
    df_existing = pd.read_parquet(save_path)
    # seen IDs to avoid duplicates (works even if timestamps shift)
    seen_ids = set(df_existing.get("reviewId", pd.Series(dtype=str)).dropna().unique())
    # latest known date per app to allow early stop
    latest_by_app = df_existing.groupby("app_name")["date"].max().to_dict()

    new_rows = []
    
    # ---- Scrape only what's new ----
    for app_name, app_id in apps.items():
        start = time.time()

        # the newest date we already have for this app; default very old
        latest_date = latest_by_app.get(app_name).tz_convert("UTC")

        token, keep_fetching, pages = None, True, 0

        while keep_fetching:
            batch, token = reviews(
                app_id,
                lang="en",
                country="gb",
                sort=Sort.NEWEST,
                count=200,
                continuation_token=token,
            )
            pages += 1

            for r in batch:
                rid, rdate = r["reviewId"], r["at"].astimezone(timezone.utc)
                if rdate <= latest_date:
                    keep_fetching = False
                    break
                if rid in seen_ids:
                    continue

                new_rows.append({
                    "app_name": app_name,
                    "app_id": app_id,
                    "reviewId": rid,
                    "user_name": r.get("userName"),
                    "score": r.get("score"),
                    "text": r.get("content"),
                    "date": pd.Timestamp(rdate),
                    "thumbs_up": r.get("thumbsUpCount"),
                    "Reply": r.get("replyContent"),
                    "Reply_Date": r.get("repliedAt"),
                    "App_Version": r.get("appVersion"),
                })
                seen_ids.add(rid)

                if token is None:
                    break  # no more pages

        print(f"✅ {app_name}: +{sum(1 for x in new_rows if x['app_name']==app_name)} new | {pages} page(s) in {time.time()-start:.2f}s")
        time.sleep(1.0)

    df_out = df_existing

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_out = (pd.concat([df_existing, df_new], ignore_index=True)
                    .drop_duplicates(subset=["reviewId"])
                    .reset_index(drop=True))
        df_out.to_parquet(save_path, index=False)
        print(f"✅ Saved {len(df_new)} new reviews (total {len(df_out)}) → {save_path}")
    else:
        print("ℹ️ No new reviews found.")

    return df_out         