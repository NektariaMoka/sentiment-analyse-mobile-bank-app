import sys
import os

# Add the banking_research directory to the Python path so it can find the 'src' folder
sys.path.append(os.path.abspath("../.."))

import pandas as pd
from pathlib import Path
from src.apps import BANKING_APPS, COUNTRY_CODES
from src.scraper import scrape_all_apps
from src.cleaning import clean_reviews

# Setup folders relative to where the script is
base_dir = Path(__file__).parent / "../../"
(base_dir / "data/raw").mkdir(parents=True, exist_ok=True)
(base_dir / "data/cleaned").mkdir(parents=True, exist_ok=True)

def run_albania():
    print("Starting scraping for ALBANIA only...")
    
    # Selection of Albania
    albania_dict = BANKING_APPS["Albania"]
    
    for bank_name, app_id in albania_dict.items():
        print(f"\nProcessing {bank_name}...")
        
        # 1. Scrape the data
        single_bank_dict = {"Albania": {bank_name: app_id}}
        all_dfs = scrape_all_apps(
            apps=single_bank_dict,
            country_codes=COUNTRY_CODES,
            count=1000,
        )

        if not all_dfs:
            print(f"No data scraped for {bank_name}. Skipping...")
            continue

        bank_df = all_dfs[0]

        # 2. Clean the data
        print(f"Cleaning data for {bank_name}...")
        clean_df = clean_reviews(bank_df)

        if clean_df.empty:
            print(f"Clean dataset for {bank_name} is empty after filtering.")
            continue

        # 3. Save the final file
        # File name: albania_bankname_clean.csv
        safe_bank_name = bank_name.lower()
        output_file = base_dir / f"data/cleaned/albania_{safe_bank_name}_clean.csv"
        
        # Ensure reviewId and other requested columns are first
        cols = ["reviewId", "review", "rating", "review_date", "app_version"]
        existing_cols = [c for c in cols if c in clean_df.columns]
        other_cols = [c for c in clean_df.columns if c not in existing_cols]
        clean_df = clean_df[existing_cols + other_cols]
        
        # Rename to exactly what you requested
        clean_df = clean_df.rename(columns={
            "review": "content",
            "rating": "score",
            "review_date": "at",
            "app_version": "reviewCreatedVersion"
        })
        
        # Final column filter as requested: reviewId, content, score, reviewCreatedVersion, at
        final_cols = ["reviewId", "content", "score", "reviewCreatedVersion", "at"]
        existing_final = [c for c in final_cols if c in clean_df.columns]
        clean_df = clean_df[existing_final]

        clean_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"SUCCESS! Saved cleaned file to: {output_file}")
        print(f"Total reviews saved for {bank_name}: {len(clean_df)}")

if __name__ == "__main__":
    run_albania()
