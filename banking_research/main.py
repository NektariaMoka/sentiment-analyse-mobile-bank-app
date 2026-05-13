import pandas as pd
from pathlib import Path

from src.apps import BANKING_APPS, COUNTRY_CODES
from src.scraper import scrape_all_apps
from src.cleaning import clean_reviews


Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/cleaned").mkdir(parents=True, exist_ok=True)


def main():
    print("Starting scraping...")

    # Scrape for each country and save cleaned files separately
    for country in BANKING_APPS.keys():
        print(f"\nProcessing {country}...")
        
        # We still want to scrape them, but we'll clean and save them one by one
        for bank_name, app_id in BANKING_APPS[country].items():
            print(f"\nScraping {bank_name} ({app_id})...")
            
            # Use a temporary dict for one bank to use existing scrape_all_apps logic 
            # or just call scrape_app_reviews directly. 
            # scrape_all_apps handles language and folders, so let's stick to it for consistency
            single_bank_dict = {country: {bank_name: app_id}}
            
            all_dfs = scrape_all_apps(
                apps=single_bank_dict,
                country_codes=COUNTRY_CODES,
                count=10000,
            )

            if not all_dfs:
                print(f"No data scraped for {bank_name}. Skipping...")
                continue

            bank_df = all_dfs[0]

            print(f"Cleaning data for {bank_name}...")
            clean_df = clean_reviews(bank_df)

            if clean_df.empty:
                print(f"Clean dataset for {bank_name} is empty.")
                continue

            # Output file name: country_bank_clean.csv
            output_file = f"data/cleaned/{country.lower()}_{bank_name.lower()}_clean.csv"
            clean_df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()