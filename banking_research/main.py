import pandas as pd
from pathlib import Path

from src.apps import BANKING_APPS, COUNTRY_CODES
from src.scraper import scrape_all_apps
from src.cleaning import clean_reviews


Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/cleaned").mkdir(parents=True, exist_ok=True)


def main():
    print("Starting scraping...")

    all_dfs = scrape_all_apps(
        apps=BANKING_APPS,
        country_codes=COUNTRY_CODES,
        count=10000,
    )

    if not all_dfs:
        print("No data scraped. Check app IDs or scraper settings.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    print("Merged columns:")
    print(merged_df.columns.tolist())

    print("Cleaning data...")
    clean_df = clean_reviews(merged_df)

    print("Clean columns:")
    print(clean_df.columns.tolist())

    if clean_df.empty:
        print("Clean dataset is empty.")
        return

    clean_df.to_csv(
        "data/cleaned/greece_banking_reviews_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved: data/cleaned/greece_banking_reviews_clean.csv")

    print("\nReviews per bank:")
    if "bank" in clean_df.columns:
        print(clean_df["bank"].value_counts())

    print("\nRatings:")
    if "rating" in clean_df.columns:
        print(clean_df["rating"].value_counts())
    else:
        print("No rating column found.")


if __name__ == "__main__":
    main()