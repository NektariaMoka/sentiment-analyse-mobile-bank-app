import time
import requests
import pandas as pd
from pathlib import Path
from google_play_scraper import reviews, Sort


def scrape_app_reviews(app_id, country_code, count=3000, languages=["en"]):
    all_results = []
    seen_review_ids = set()

    for lang in languages:
        print(f"  Scraping Google Play language: {lang}")
        try:
            result, _ = reviews(
                app_id,
                lang=lang,
                country=country_code,
                sort=Sort.NEWEST,
                count=count,
            )
            for r in result:
                if r["reviewId"] not in seen_review_ids:
                    seen_review_ids.add(r["reviewId"])
                    # Standardize format for combined analysis
                    r["source"] = "google_play"
                    all_results.append(r)
        except Exception as e:
            print(f"    Error scraping {lang}: {e}")

    return pd.DataFrame(all_results)


def scrape_app_store_reviews(app_id, country="al"):
    """
    Scrapes reviews using the iTunes RSS feed.
    """
    print(f"  Scraping App Store (RSS): {app_id} ({country})")
    all_reviews = []
    
    # iTunes RSS feed for customer reviews
    # Page can be 1 to 10. Actually, it can go up to 10 pages maximum for RSS.
    for page in range(1, 11):
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                break
                
            data = response.json()
            feed = data.get("feed", {})
            entries = feed.get("entry", [])
            
            if not entries:
                break
                
            # If only one entry, it might be a dict instead of list
            if isinstance(entries, dict):
                entries = [entries]
                
            for entry in entries:
                # The first entry is often the app metadata, not a review
                if "im:name" in entry:
                    continue
                    
                review = {
                    "reviewId": entry.get("id", {}).get("label"),
                    "userName": entry.get("author", {}).get("name", {}).get("label"),
                    "content": entry.get("content", {}).get("label"),
                    "score": int(entry.get("im:rating", {}).get("label", 0)),
                    "at": entry.get("updated", {}).get("label"),
                    "reviewCreatedVersion": entry.get("im:version", {}).get("label"),
                    "review_title": entry.get("title", {}).get("label"),
                    "source": "app_store"
                }
                all_reviews.append(review)
                
            # Small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error fetching App Store page {page}: {e}")
            break
            
    return pd.DataFrame(all_reviews)


def scrape_all_apps(apps, country_codes, count=3000):
    all_dfs = []

    for country, country_apps in apps.items():
        print(f"\n===== Scraping {country} =====")
        # Languages for Google Play
        if country == "Greece":
            langs = ["el", "en"]
        elif country == "Albania":
            # Expanding languages to cover more potential reviews
            langs = [
                "sq", "en", "it"
            ]
        else:
            langs = ["en"]

        for bank_name, config in country_apps.items():
            bank_dfs = []
            
            # 1. Scrape Google Play
            if "play_id" in config:
                print(f"Scraping Google Play for {bank_name}: {config['play_id']}")
                play_df = scrape_app_reviews(
                    app_id=config['play_id'],
                    country_code=country_codes[country],
                    count=count,
                    languages=langs,
                )
                if not play_df.empty:
                    bank_dfs.append(play_df)

            # 2. Scrape App Store
            if "apple_id" in config:
                print(f"Scraping App Store for {bank_name}: {config['apple_id']}")
                # Scrape primary country
                apple_df = scrape_app_store_reviews(
                    app_id=config['apple_id'],
                    country=country_codes[country]
                )
                if not apple_df.empty:
                    bank_dfs.append(apple_df)
                
                # Also try 'us' store for Albania as it's common
                if country == "Albania":
                    print(f"  Scraping App Store (US) for {bank_name}...")
                    apple_us_df = scrape_app_store_reviews(
                        app_id=config['apple_id'],
                        country="us"
                    )
                    if not apple_us_df.empty:
                        bank_dfs.append(apple_us_df)

            if not bank_dfs:
                continue

            df = pd.concat(bank_dfs, ignore_index=True)
            # Ensure the data/raw directory exists
            raw_dir = Path("data/raw")
            if not raw_dir.parent.exists() and Path("banking_research").exists():
                raw_dir = Path("banking_research/data/raw")
            
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"{country}_{bank_name}.csv"
            df.to_csv(raw_file, index=False)

            all_dfs.append(df)
            print(f"Saved {len(df)} reviews to {raw_file}")
            time.sleep(1)

    return all_dfs