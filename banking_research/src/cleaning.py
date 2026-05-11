import pandas as pd


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

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["review"], inplace=True)

    df = df[df["review"].astype(str).str.len() > 5]

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        df["year"] = df["review_date"].dt.year

    df["review_length"] = df["review"].astype(str).apply(len)

    df.reset_index(drop=True, inplace=True)

    return df