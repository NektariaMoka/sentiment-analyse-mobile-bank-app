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
    
    for bank_name, app_id in albania_dict.items():
        print(f"\nProcessing {bank_name}...")
        
        # 1. Scrape the data
        single_bank_dict = {"Albania": {bank_name: app_id}}
        all_dfs = scrape_all_apps(
            apps=single_bank_dict,
            country_codes=COUNTRY_CODES,
            count=10000,
        )

        if not all_dfs:
            print(f"No data scraped for {bank_name}. Skipping...")
            continue

        bank_df = all_dfs[0]

        # 3. Format and save the final file
        # Requested columns:
        # Package Name, App Version Code, App Version Name, Reviewer Language, Device, 
        # Review Submit Date and Time, Review Submit Millis Since Epoch, 
        # Review Last Update Date and Time, Review Last Update Millis Since Epoch, 
        # Star Rating, Review Title, Review Text, 
        # Developer Reply Date and Time, Developer Reply Millis Since Epoch, Developer Reply Text, 
        # Review Link

        # Ensure 'at' is datetime and convert to UTC to avoid mixed timezone issues
        bank_df["at"] = pd.to_datetime(bank_df["at"], errors="coerce", utc=True)
        
        # Helper to get millis
        def to_millis(dt):
            try:
                if pd.isna(dt): return ""
                return int(dt.timestamp() * 1000)
            except:
                return ""

        formatted_df = pd.DataFrame()
        formatted_df["Package Name"] = bank_df["app_id"]
        formatted_df["App Version Code"] = "" # Not directly available from scraper
        formatted_df["App Version Name"] = bank_df["reviewCreatedVersion"]
        formatted_df["Reviewer Language"] = "en" # We scraped en
        formatted_df["Device"] = "" # Not available from scraper
        
        formatted_df["Review Submit Date and Time"] = bank_df["at"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        formatted_df["Review Submit Millis Since Epoch"] = bank_df["at"].apply(to_millis)
        
        # Google Play scraper 'at' is the last update time
        formatted_df["Review Last Update Date and Time"] = formatted_df["Review Submit Date and Time"]
        formatted_df["Review Last Update Millis Since Epoch"] = formatted_df["Review Submit Millis Since Epoch"]
        
        formatted_df["Star Rating"] = bank_df["score"]
        formatted_df["Review Title"] = "" # Google Play reviews usually don't have titles in this scraper
        formatted_df["Review Text"] = bank_df["content"]
        
        # Developer Reply
        if "repliedAt" in bank_df.columns:
            bank_df["repliedAt"] = pd.to_datetime(bank_df["repliedAt"], errors="coerce")
            formatted_df["Developer Reply Date and Time"] = bank_df["repliedAt"].dt.strftime('%Y-%m-%dT%H:%M:%SZ').fillna("")
            formatted_df["Developer Reply Millis Since Epoch"] = bank_df["repliedAt"].apply(to_millis)
        else:
            formatted_df["Developer Reply Date and Time"] = ""
            formatted_df["Developer Reply Millis Since Epoch"] = ""
            
        formatted_df["Developer Reply Text"] = bank_df.get("replyContent", "")
        formatted_df["Review Link"] = "" # Could be constructed but often empty in these exports

        # File name: albania_bankname_clean.csv
        safe_bank_name = bank_name.lower()
        output_file = base_dir / f"data/cleaned/albania_{safe_bank_name}_clean.csv"
        
        formatted_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"SUCCESS! Saved formatted file to: {output_file}")
        print(f"Total reviews saved for {bank_name}: {len(formatted_df)}")

if __name__ == "__main__":
    run_albania()
