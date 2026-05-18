import pandas as pd
from pathlib import Path

# Instead of one single file, we now look for multiple bank files
CLEANED_DIR = Path("./data/cleaned")

albania_banks = [
    "bkt_smart", "credins_online", "raiffeisen_on", 
     "otp_bank", "intesa_sanpaolo"
]

found_any = False

for bank in albania_banks:
    DATA_PATH = CLEANED_DIR / f"albania_{bank}_clean.csv"
    
    if not DATA_PATH.exists():
        continue
    
    found_any = True
    print(f"\n========================================")
    print(f"ANALYZING BANK: {bank.upper()}")
    print(f"========================================")
    
    df = pd.read_csv(DATA_PATH)

    # Columns are already renamed in the new run_albania_only.py
    # requested_cols = ["reviewId", "content", "score", "reviewCreatedVersion", "at"]
    
    print("\n===== BASIC INFO =====")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    
    print("\n===== DATA PREVIEW (First 5 rows) =====")
    print(df.head())

    print("\n===== RATINGS DISTRIBUTION =====")
    if "Star Rating" in df.columns:
        print(df["Star Rating"].value_counts().sort_index())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DATE RANGE =====")
    if "Review Submit Date" in df.columns:
        df["Review Submit Date"] = pd.to_datetime(df["Review Submit Date"], errors="coerce")
        print("Oldest review:", df["Review Submit Date"].min().date())
        print("Newest review:", df["Review Submit Date"].max().date())

    print("\n===== SAMPLE 1-STAR REVIEWS =====")
    if "Star Rating" in df.columns and "Review Text" in df.columns:
        one_star = df[df["Star Rating"] == 1][["Star Rating", "Review Text"]].sample(
            min(5, len(df[df["Star Rating"] == 1])),
            random_state=42
        )
        print(one_star.to_string(index=False))

if not found_any:
    print(f"No cleaned Albanian bank files found in {CLEANED_DIR}.")
    print("Please run run_albania_only.py first.")

Path("../../outputs").mkdir(exist_ok=True)
# The summary logic would need a loop over all banks to recreate a cross-bank summary
# For now, I'll stop here as the main request was about separate files.
