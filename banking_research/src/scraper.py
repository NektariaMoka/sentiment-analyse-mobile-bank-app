import time
import pandas as pd
from google_play_scraper import reviews, Sort


def scrape_app_reviews(app_id, country_code, count=3000):
    result, _ = reviews(
        app_id,
        lang="el",
        country=country_code,
        sort=Sort.NEWEST,
        count=count,
    )

    return pd.DataFrame(result)


def scrape_all_apps(apps, country_codes, count=3000):
    all_dfs = []

    for country, country_apps in apps.items():
        print(f"\n===== Scraping {country} =====")

        for bank_name, app_id in country_apps.items():
            print(f"Scraping {bank_name}: {app_id}")

            try:
                df = scrape_app_reviews(
                    app_id=app_id,
                    country_code=country_codes[country],
                    count=count,
                )

                df["country"] = country
                df["bank"] = bank_name
                df["app_id"] = app_id

                raw_file = f"data/raw/{country}_{bank_name}.csv"
                df.to_csv(raw_file, index=False)

                all_dfs.append(df)

                print(f"Saved {len(df)} reviews to {raw_file}")

                time.sleep(2)

            except Exception as error:
                print(f"ERROR scraping {bank_name}: {error}")

    return all_dfs