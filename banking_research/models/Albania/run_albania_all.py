import sys
import os

# Add the banking_research directory to the Python path so it can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from pathlib import Path
from src.apps import BANKING_APPS, COUNTRY_CODES
from src.scraper import scrape_all_apps
from src.cleaning import clean_reviews

# Setup folders relative to where the script is
base_dir = Path(__file__).parent
(base_dir / "data/raw").mkdir(parents=True, exist_ok=True)
(base_dir / "data/cleaned").mkdir(parents=True, exist_ok=True)

def run_albania():
    print("Starting scraping for ALBANIA only...")
    
    # Selection of Albania
    albania_dict = BANKING_APPS["Albania"]
    
    all_formatted_dfs = []
    
    for bank_name, config in albania_dict.items():
        print(f"\nProcessing {bank_name}...")
        
        # 1. Scrape the data
        single_bank_dict = {"Albania": {bank_name: config}}
        all_dfs = scrape_all_apps(
            apps=single_bank_dict,
            country_codes=COUNTRY_CODES,
            count=10000,
        )

        if not all_dfs:
            print(f"No data scraped for {bank_name}. Skipping...")
            continue

        bank_df = all_dfs[0]

        # 2. No filtering (ALL data)
        
        # 3. Format
        bank_df = bank_df.copy()
        bank_df["at"] = pd.to_datetime(bank_df["at"], errors="coerce", utc=True)
        
        def to_millis(dt):
            try:
                if pd.isna(dt): return ""
                return int(dt.timestamp() * 1000)
            except:
                return ""

        formatted_df = pd.DataFrame()
        formatted_df["Package Name"] = bank_df.apply(lambda x: config.get("play_id") if x["source"] == "google_play" else config.get("apple_id"), axis=1)
        formatted_df["App Version Code"] = ""
        formatted_df["App Version Name"] = bank_df["reviewCreatedVersion"]
        formatted_df["Reviewer Language"] = "en"
        formatted_df["Device"] = ""
        
        formatted_df["Review Submit Date and Time"] = bank_df["at"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        formatted_df["Review Submit Millis Since Epoch"] = bank_df["at"].apply(to_millis)
        
        formatted_df["Review Last Update Date and Time"] = formatted_df["Review Submit Date and Time"]
        formatted_df["Review Last Update Millis Since Epoch"] = formatted_df["Review Submit Millis Since Epoch"]
        
        formatted_df["Star Rating"] = bank_df["score"]
        formatted_df["Review Title"] = bank_df.get("review_title", "")
        formatted_df["Review Text"] = bank_df["content"]
        
        if "repliedAt" in bank_df.columns:
            bank_df["repliedAt"] = pd.to_datetime(bank_df["repliedAt"], errors="coerce")
            formatted_df["Developer Reply Date and Time"] = bank_df["repliedAt"].dt.strftime('%Y-%m-%dT%H:%M:%SZ').fillna("")
            formatted_df["Developer Reply Millis Since Epoch"] = bank_df["repliedAt"].apply(to_millis)
        else:
            formatted_df["Developer Reply Date and Time"] = ""
            formatted_df["Developer Reply Millis Since Epoch"] = ""
            
        formatted_df["Developer Reply Text"] = bank_df.get("replyContent", "")
        formatted_df["Review Link"] = ""

        all_formatted_dfs.append(formatted_df)
        print(f"Added {len(formatted_df)} reviews for {bank_name} to collection.")

    if all_formatted_dfs:
        final_df = pd.concat(all_formatted_dfs, ignore_index=True)
        output_file = base_dir / "data/cleaned/albania_all_banks_clean.csv"
        final_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\nSUCCESS! Saved consolidated file to: {output_file}")
        print(f"Total reviews saved: {len(final_df)}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    run_albania()
