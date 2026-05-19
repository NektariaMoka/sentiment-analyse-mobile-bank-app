import re
import pandas as pd
from pathlib import Path
from transformers import pipeline

# =========================
# PATHS
# =========================

base_dir = Path(__file__).parent
data_dir = base_dir / "data/cleaned"
output_dir = base_dir / "data/bert_only"
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT CLEANING
# =========================

ALBANIAN_STOPWORDS = {
    "a", "apo", "asnje", "ata", "ato", "ca", "deri", "dhe", "do", "e", "i", "jam",
    "jane", "jemi", "jeni", "ju", "juaj", "kam", "kaq", "ke", "kemi", "kete",
    "me", "mu", "ne", "nese", "nje", "nuk", "pa", "pas", "pasi", "per", "prej",
    "qe", "sa", "se", "sec", "si", "saj", "te", "ti", "tek", "tij", "tone",
    "tuaj", "ty", "tyre", "une", "vec"
}

def normalize_albanian(text, remove_stopwords=True):
    text = str(text).lower()
    text = text.replace("ë", "e").replace("ç", "c")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in ALBANIAN_STOPWORDS]
        text = " ".join(words)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# BERT SENTIMENT MODEL
# =========================

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None
)

def parse_pipe_results(pipe_results):
    """
    Parses the raw results from the sentiment pipeline into categorical labels and scores.
    """
    batch_data = []
    for results in pipe_results:
        # nlptown labels: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
        probs = {r["label"]: r["score"] for r in results}
        
        neg = probs.get("1 star", 0) + probs.get("2 stars", 0)
        neu = probs.get("3 stars", 0)
        pos = probs.get("4 stars", 0) + probs.get("5 stars", 0)
        
        if pos > neg and pos > neu:
            label = "Positive"
            score = pos
        elif neg > pos and neg > neu:
            label = "Negative"
            score = neg
        else:
            label = "Neutral"
            score = neu
            
        batch_data.append([
            label, 
            round(score, 4), 
            round(pos, 4), 
            round(neu, 4), 
            round(neg, 4)
        ])
    return batch_data

# =========================
# MAIN PROCESS
# =========================

def process_file(file_path):
    print(f"Processing {file_path.name}...")

    df = pd.read_csv(file_path)

    text_col = "Review Text" if "Review Text" in df.columns else "review"

    df["clean_text"] = df[text_col].apply(lambda x: normalize_albanian(x, remove_stopwords=True))

    df = df[df["clean_text"].str.strip() != ""].copy()

    # -------------------------
    # BERT SENTIMENT (BATCH PROCESSING)
    # -------------------------

    # Convert column to list for batching
    texts = df[text_col].astype(str).tolist()

    # batch_size=16 is a safe balance for memory and performance
    # truncation=True ensures we don't exceed BERT's 512 token limit
    raw_results = sentiment_pipe(texts, batch_size=16, truncation=True)
    
    sentiment_data = parse_pipe_results(raw_results)

    df[
        [
            "Predicted_Sentiment",
            "Probability_Score",
            "Positive_Prob",
            "Neutral_Prob",
            "Negative_Prob"
        ]
    ] = pd.DataFrame(sentiment_data, index=df.index)

    final_df = df.copy()

    # -------------------------
    # SAVE OUTPUT
    # -------------------------

    # Extract the bank name from the filename (e.g., from 'albania_bkt_smart_clean.csv' extract 'bkt_smart')
    bank_name = file_path.name.replace("albania_", "").replace("_clean.csv", "")
    output_csv = output_dir / f"bert_{bank_name}.csv"

    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved CSV: {output_csv}")

# =========================
# RUN ALL ALBANIA FILES
# =========================

def main():
    files = list(data_dir.glob("albania_*_clean.csv"))

    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()
