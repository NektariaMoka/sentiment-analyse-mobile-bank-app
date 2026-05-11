import pandas as pd
from pathlib import Path

DATA_PATH = "data/cleaned/greece_banking_reviews_clean.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== BASIC INFO =====")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n===== REVIEWS PER BANK =====")
print(df["bank"].value_counts())

print("\n===== RATINGS DISTRIBUTION =====")
print(df["rating"].value_counts().sort_index())

print("\n===== AVERAGE RATING PER BANK =====")
print(df.groupby("bank")["rating"].mean().sort_values(ascending=False))

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATE RANGE =====")
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
print("Oldest review:", df["review_date"].min())
print("Newest review:", df["review_date"].max())

print("\n===== REVIEWS PER YEAR =====")
print(df["review_date"].dt.year.value_counts().sort_index())

print("\n===== REVIEW LENGTH =====")
print(df["review_length"].describe())

print("\n===== SAMPLE 1-STAR REVIEWS =====")
one_star = df[df["rating"] == 1][["bank", "rating", "review"]].sample(
    min(10, len(df[df["rating"] == 1])),
    random_state=42
)
print(one_star.to_string(index=False))

print("\n===== SAMPLE 5-STAR REVIEWS =====")
five_star = df[df["rating"] == 5][["bank", "rating", "review"]].sample(
    min(10, len(df[df["rating"] == 5])),
    random_state=42
)
print(five_star.to_string(index=False))

Path("outputs").mkdir(exist_ok=True)

summary = df.groupby("bank").agg(
    reviews=("review", "count"),
    average_rating=("rating", "mean"),
    median_rating=("rating", "median"),
    min_date=("review_date", "min"),
    max_date=("review_date", "max"),
)

summary.to_csv("outputs/greece_dataset_summary.csv", encoding="utf-8-sig")

print("\nSaved summary to outputs/greece_dataset_summary.csv")