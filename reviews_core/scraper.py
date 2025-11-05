import pandas as pd
from google_play_scraper import  reviews, Sort
from pathlib import Path
import time
from datetime import timezone

def scrape_new_reviews():
    # List of APPs' packages to be scraped
    apps = {
        "Santander": "uk.co.santander.santanderUK",
        "HSBC": "uk.co.hsbc.hsbcukmobilebanking",
        "Lloyds": "com.grppl.android.shell.CMBlloydsTSB73",
        "Barclays": "com.barclays.android.barclaysmobilebanking",
        "Revolut": "com.revolut.revolut",
        "Monzo": "co.uk.getmondo"
    }

    # existing full dataset
    existing_path = Path("../assets/dfs_pipeline/df_raw.parquet")
    df_raw = pd.read_parquet(existing_path)

    # Path to save new reviews
    new_path = Path("../assets/dfs_pipeline/new_df_raw.parquet")

    # seen IDs to avoid duplicates
    seen_ids = set(df_raw.get("reviewId", pd.Series(dtype=str)).dropna().unique())

    # latest known date per app to allow early stop
    latest_by_app = df_raw.groupby("app_name")["date"].max().to_dict()

    new_rows = []

    # ---- Scrape only what's new ----
    for app_name, app_id in apps.items():
        start = time.time()

        # the newest date we already have for this app
        latest_date = latest_by_app.get(app_name)

        # defend in case some app had no rows before
        if latest_date is None:
            # make it very old so we pull everything
            latest_date = pd.Timestamp("2000-01-01", tz="UTC")
        else:
            # ensure UTC like you had
            latest_date = latest_date.tz_convert("UTC")

        token, keep_fetching, pages = None, True, 0
        app_new_count_before = len(new_rows)

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
                rid = r["reviewId"]
                rdate = r["at"].astimezone(timezone.utc)

                # stop as soon as we hit an old review
                if rdate <= latest_date:
                    keep_fetching = False
                    break

                # skip if we already had this reviewId
                if rid in seen_ids:
                    continue

                # fetch relevant fields
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
                # no more pages from store
                break

        app_new_count_after = len(new_rows)
        app_added = app_new_count_after - app_new_count_before
        print(f"✅ {app_name}: +{app_added} new | {pages} page(s) in {time.time()-start:.2f}s")

        time.sleep(1.0)

    # ---- Build and save Dataframe with new reviews and update existing ----
    if new_rows:
        new_df_raw = pd.DataFrame(new_rows)
        # add column with scrape date
        new_df_raw["scrape_date"] = pd.Timestamp.now(tz=timezone.utc)
        # save ONLY the new rows
        new_df_raw.to_parquet(new_path, index=False)
        print(f"✅ Saved {len(new_df_raw)} new reviews → {new_path}")

        # update existing full dataset
        df_raw = pd.concat([df_raw, new_df_raw], ignore_index=True)
        df_raw.to_parquet(existing_path, index=False)
        print(f"✅ Updated existing df_raw with new reviews. New shape: {df_raw.shape}")

    else:
        print("ℹ️ No new reviews found. Saved empty new_df_raw.")

    return new_df_raw