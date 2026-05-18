import re
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic

# =========================
# PATHS
# =========================

base_dir = Path(__file__).parent
data_dir = base_dir / "data/cleaned"
output_dir = base_dir / "data/bert_topic"
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT CLEANING
# =========================

def normalize_albanian(text):
    text = str(text).lower()
    text = text.replace("ë", "e").replace("ç", "c")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# MAIN PROCESS
# =========================

def process_file(file_path):
    print(f"Processing {file_path.name}...")

    df = pd.read_csv(file_path)

    text_col = "Review Text" if "Review Text" in df.columns else "review"

    df["clean_text"] = df[text_col].astype(str).apply(normalize_albanian)
    df = df[df["clean_text"].str.strip() != ""].copy()

    docs = df["clean_text"].tolist()

    # Simple BERTopic model with default settings as requested
    topic_model = BERTopic(
        language="multilingual",
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    # ==============================
    # 5 TOPIC PROBABILITIES
    # ==============================
    topic_prob_df = pd.DataFrame(probs)
    topic_prob_df.columns = [
        f"topic_{i}" for i in range(topic_prob_df.shape[1])
    ]
    topic_prob_df = topic_prob_df.replace([np.inf, -np.inf], np.nan)
    topic_prob_df = topic_prob_df.fillna(0)

    # Concatenate with original data
    df_model = pd.concat(
        [df.reset_index(drop=True), topic_prob_df],
        axis=1
    )

    df_model["topic"] = topics
    df_model["review_length"] = df_model["clean_text"].apply(
        lambda x: len(x.split())
    )

    # Rishiko topikët dhe etiketo manualisht
    topic_info_check = topic_model.get_topic_info()
    print("\nKontrollo topikët dhe etiketo manualisht:")
    for t in topic_info_check["Topic"]:
        if t == -1:
            continue
        kws = topic_model.get_topic(t)
        if kws and kws is not False:
            print(f"  Topic {t}: {[w for w, _ in kws[:5]]}")

    # *** ETIKETO SIPAS INSPEKTIMIT ***
    # Ndrysho map-in pas kontrollit të keywords
    TOPIC_LABELS = {
        "topic_0": "Ease_of_Use",           # app, great, easy
        "topic_1": "Technical_Issues",      # nuk, hapet, punon
        "topic_2": "General_Satisfaction",  # shume, mire, nice
    }
    
    # Rename only existing columns to avoid errors if there are fewer topics
    existing_labels = {k: v for k, v in TOPIC_LABELS.items() if k in topic_prob_df.columns}
    topic_prob_df_labeled = topic_prob_df.rename(columns=existing_labels)

    # Rindërtojmë df_model me emrat e saktë (ose të paktën ata që u gjetën)
    df_model = pd.concat(
        [df.reset_index(drop=True), topic_prob_df_labeled],
        axis=1
    )
    df_model["topic"] = topics
    df_model["review_length"] = df_model["clean_text"].apply(
        lambda x: len(x.split())
    )

    # Save output
    # Extract the bank name from the filename (e.g., from 'albania_bkt_smart_clean.csv' extract 'bkt_smart')
    bank_name = file_path.name.replace("albania_", "").replace("_clean.csv", "")
    output_csv = output_dir / f"bert_topic_{bank_name}.csv"

    df_model.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")

def main():
    files = list(data_dir.glob("albania_*_clean.csv"))
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()