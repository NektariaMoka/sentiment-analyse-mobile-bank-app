import pandas as pd
import re
import string


def clean_reviews(df):
    print("Columns before cleaning:")
    print(df.columns.tolist())

    if df.empty:
        print("Dataframe is empty. No reviews were scraped.")
        return df

    # Rename only if these columns exist
    df = df.rename(
        columns={
            "content": "review",
            "score": "rating",
            "at": "review_date",
            "appVersion": "app_version",
            "thumbsUpCount": "helpful_votes",
            "replyContent": "developer_reply",
        }
    )

    print("Columns after renaming:")
    print(df.columns.tolist())

    if "review" not in df.columns:
        raise ValueError(
            "Column 'review' was not found. This probably means Google Play did not return review content."
        )

    wanted_columns = [
        "reviewId",
        "review",
        "rating",
        "review_date",
        "app_version",
        "helpful_votes",
        "developer_reply",
        "country",
        "bank",
        "app_id",
    ]

    existing_columns = [col for col in wanted_columns if col in df.columns]
    df = df[existing_columns].copy()

    # 1. Elimination of duplicate entries
    df.drop_duplicates(subset=["reviewId"], inplace=True)
    df.drop_duplicates(subset=["review"], inplace=True)

    # 2. Removal of empty reviews
    df.dropna(subset=["review"], inplace=True)
    df["review"] = df["review"].astype(str)

    # 3. Conversion of text to lowercase
    df["review"] = df["review"].str.lower()

    # 4. Removal of punctuation and special characters
    def remove_punctuation(text):
        # Keep only alphanumeric (including Albanian characters ë and ç) and spaces
        # Using regex to match anything that is NOT a word character (a-z, 0-9), Albanian chars, or whitespace
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z0-9\sëç]", " ", text)
        return text

    df["review"] = df["review"].apply(remove_punctuation)

    # 5. Clean up extra whitespace resulting from punctuation removal
    df["review"] = df["review"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 6. Removal of empty reviews (if any left)
    # Previously we filtered for length > 3, but now we keep all reviews to include ratings
    df = df[df["review"].str.len() >= 0]

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
        df["year"] = df["review_date"].dt.year

    df["review_length"] = df["review"].apply(len)

    df.reset_index(drop=True, inplace=True)

    return df