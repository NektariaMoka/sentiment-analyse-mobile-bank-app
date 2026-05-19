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

def split_into_sentences(text):
    """
    Splits text into sentences using basic punctuation markers.
    Works for Albanian as it uses standard . ! ? for sentence ending.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Split by . ! or ? followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

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
    batch_probs = []
    for results in pipe_results:
        # nlptown labels: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
        probs = {r["label"]: r["score"] for r in results}
        
        neg = probs.get("1 star", 0) + probs.get("2 stars", 0)
        neu = probs.get("3 stars", 0)
        pos = probs.get("4 stars", 0) + probs.get("5 stars", 0)
        
        batch_probs.append({
            "pos": pos,
            "neu": neu,
            "neg": neg
        })
    return batch_probs

# =========================
# MAIN PROCESS
# =========================

def process_file(file_path):
    print(f"Processing {file_path.name} (Sentence-Level Analysis)...")

    df = pd.read_csv(file_path)

    text_col = "Review Text" if "Review Text" in df.columns else "review"

    # Step 1: Split reviews into sentences
    review_sentences = []
    review_indices = []
    
    for idx, row in df.iterrows():
        text = str(row[text_col])
        sentences = split_into_sentences(text)
        if not sentences:
            sentences = [text] if text.strip() else ["Neutral"]
            
        review_sentences.extend(sentences)
        review_indices.extend([idx] * len(sentences))

    # Step 2: Batch process all sentences
    print(f"Total sentences to process: {len(review_sentences)}")
    
    # batch_size=16 is a safe balance for memory and performance
    # truncation=True ensures we don't exceed BERT's 512 token limit
    raw_results = sentiment_pipe(review_sentences, batch_size=16, truncation=True)
    sentence_probs = parse_pipe_results(raw_results)

    # Step 3: Aggregate results back to review level
    temp_df = pd.DataFrame(sentence_probs)
    temp_df['review_idx'] = review_indices
    
    # Average the probabilities for all sentences in a review
    agg_probs = temp_df.groupby('review_idx').mean()

    # Step 4: Map back to original dataframe and determine final sentiment
    def get_final_sentiment(row):
        pos, neu, neg = row['pos'], row['neu'], row['neg']
        if pos > neg and pos > neu:
            return "Positive", pos
        elif neg > pos and neg > neu:
            return "Negative", neg
        else:
            return "Neutral", neu

    # Apply classification logic to the aggregated probabilities
    final_results = agg_probs.apply(get_final_sentiment, axis=1)

    df['Predicted_Sentiment'] = [final_results[i][0] for i in df.index]
    df['Probability_Score'] = [round(final_results[i][1], 4) for i in df.index]
    df['Positive_Prob'] = [round(agg_probs.loc[i, 'pos'], 4) for i in df.index]
    df['Neutral_Prob'] = [round(agg_probs.loc[i, 'neu'], 4) for i in df.index]
    df['Negative_Prob'] = [round(agg_probs.loc[i, 'neg'], 4) for i in df.index]

    # Add clean_text for consistency (normalized full review)
    df["clean_text"] = df[text_col].apply(lambda x: normalize_albanian(x, remove_stopwords=True))

    # Filter out rows where clean_text is empty if needed, but here we keep all original rows
    # as we processed all sentences.
    
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
