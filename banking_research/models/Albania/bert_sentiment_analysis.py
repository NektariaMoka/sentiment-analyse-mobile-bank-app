import re
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def normalize_albanian(text):
    text = str(text).lower()
    text = text.replace("ë", "e").replace("ç", "c")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
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

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0, 1.0, 0.0

    try:
        results = sentiment_pipe(text[:512])[0]

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

        return label, round(score, 4), round(pos, 4), round(neu, 4), round(neg, 4)

    except Exception:
        return "Neutral", 0.0, 0.0, 1.0, 0.0

# =========================
# BERT EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def create_embeddings(texts):
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

# =========================
# MAIN PROCESS
# =========================

def process_file(file_path):
    print(f"Processing {file_path.name}...")

    df = pd.read_csv(file_path)

    text_col = "Review Text" if "Review Text" in df.columns else "review"

    df["clean_text"] = df[text_col].apply(normalize_albanian)

    df = df[df["clean_text"].str.strip() != ""].copy()

    # -------------------------
    # BERT SENTIMENT
    # -------------------------

    sentiment_results = df[text_col].apply(get_sentiment)

    df[
        [
            "Predicted_Sentiment",
            "Probability_Score",
            "Positive_Prob",
            "Neutral_Prob",
            "Negative_Prob"
        ]
    ] = pd.DataFrame(sentiment_results.tolist(), index=df.index)

    # -------------------------
    # BERT EMBEDDINGS
    # -------------------------

    docs = df["clean_text"].tolist()

    embeddings = create_embeddings(docs)

    # We no longer save embedding dimensions as columns in the CSV to keep it simple for research.
    # The embeddings are still saved separately in the .npy file.

    final_df = df.copy()

    # -------------------------
    # SAVE OUTPUT
    # -------------------------

    output_csv = output_dir / file_path.name.replace(
        "_clean.csv",
        "_bert_only.csv"
    )

    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved CSV: {output_csv}")

    # Optional: save embeddings separately
    output_npy = output_dir / file_path.name.replace(
        "_clean.csv",
        "_embeddings.npy"
    )

    np.save(output_npy, embeddings)

    print(f"Saved embeddings: {output_npy}")

# =========================
# RUN ALL ALBANIA FILES
# =========================

def main():
    files = list(data_dir.glob("albania_*_clean.csv"))

    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()
