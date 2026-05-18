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
        
        formatted_df = pd.DataFrame()
        formatted_df["Package Name"] = bank_df.apply(lambda x: config.get("play_id") if x["source"] == "google_play" else config.get("apple_id"), axis=1)
        formatted_df["App Version Code"] = ""
        formatted_df["App Version Name"] = bank_df["reviewCreatedVersion"]
        formatted_df["Reviewer Language"] = "en"
        formatted_df["Device"] = ""
        
        formatted_df["Review Submit Date"] = bank_df["at"].dt.strftime('%Y-%m-%d')
        formatted_df["Review Last Update Date"] = formatted_df["Review Submit Date"]
        
        formatted_df["Star Rating"] = bank_df["score"]
        formatted_df["Review Title"] = bank_df.get("review_title", "")
        formatted_df["Review Text"] = bank_df["content"]
        
        if "repliedAt" in bank_df.columns:
            bank_df["repliedAt"] = pd.to_datetime(bank_df["repliedAt"], errors="coerce")
            formatted_df["Developer Reply Date"] = bank_df["repliedAt"].dt.strftime('%Y-%m-%d').fillna("")
        else:
            formatted_df["Developer Reply Date"] = ""
            
        formatted_df["Developer Reply Text"] = bank_df.get("replyContent", "")
        formatted_df["Review Link"] = ""

        # Save individual bank file
        output_file = base_dir / f"data/cleaned/albania_{bank_name.lower()}_clean.csv"
        formatted_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"SUCCESS! Saved individual file to: {output_file}")
        print(f"Added {len(formatted_df)} reviews for {bank_name}.")

if __name__ == "__main__":
    run_albania()
