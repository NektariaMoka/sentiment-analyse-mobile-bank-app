import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Setup paths relative to the script
base_dir = Path(__file__).parent
data_dir = base_dir / "data/cleaned"
output_dir = base_dir / "data/sentiment"
output_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading sentiment model...")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_model = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

def get_sentiment_details(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "Predicted_Sentiment": "Neutral",
            "Positive_Prob": 0.0,
            "Neutral_Prob": 1.0,
            "Negative_Prob": 0.0,
            "Tokens": [],
            "Token_Vectors": []
        }
    try:
        # Use first 512 characters as BERT has a limit
        truncated_text = text[:512]
        
        # 1. Get Sentiment
        results = sentiment_model(truncated_text)[0]
        
        # 2. Get Vector (Embeddings)
        # We'll extract embeddings for ALL tokens in the sequence
        inputs = tokenizer(truncated_text, return_tensors="pt", truncation=True, max_length=512)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # last_hidden_state shape: [batch_size, sequence_length, hidden_size]
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            
            # Extract vectors for all tokens in the batch (batch size 1)
            # result is a list of lists (one list of 768 floats per token)
            token_vectors = last_hidden_state[0].cpu().numpy().tolist()
            # Round for readability and space
            token_vectors = [[round(val, 6) for val in vec] for vec in token_vectors]
        
        # Mapping: 
        # 1-2 stars -> Negative
        # 3 stars -> Neutral
        # 4-5 stars -> Positive
        
        probs = {res["label"]: res["score"] for res in results}
        
        neg_prob = probs.get("1 star", 0) + probs.get("2 stars", 0)
        neu_prob = probs.get("3 stars", 0)
        pos_prob = probs.get("4 stars", 0) + probs.get("5 stars", 0)
        
        if pos_prob > neu_prob and pos_prob > neg_prob:
            pred = "Positive"
        elif neg_prob > neu_prob and neg_prob > pos_prob:
            pred = "Negative"
        else:
            pred = "Neutral"
            
        return {
            "Predicted_Sentiment": pred,
            "Positive_Prob": round(pos_prob, 4),
            "Neutral_Prob": round(neu_prob, 4),
            "Negative_Prob": round(neg_prob, 4),
            "Tokens": tokens,
            "Token_Vectors": token_vectors
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "Predicted_Sentiment": "Neutral",
            "Positive_Prob": 0.0,
            "Neutral_Prob": 1.0,
            "Negative_Prob": 0.0,
            "Tokens": [],
            "Token_Vectors": []
        }

def process_files():
    # Find all cleaned CSV files for Albania
    files = list(data_dir.glob("albania_*_clean.csv"))
    
    if not files:
        print(f"No cleaned files found in {data_dir}")
        return

    for file_path in files:
        print(f"\nProcessing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        # Map columns to match user's requested format
        # User requested: Review_ID, Review_Text, Rating, Predicted_Sentiment, Positive_Prob, Neutral_Prob, Negative_Prob
        
        # 1. Review_Text
        text_col = "Review Text"
        if text_col not in df.columns:
            if 'review' in df.columns:
                text_col = 'review'
            else:
                print(f"Could not find text column in {file_path.name}. Skipping...")
                continue
        
        # 2. Rating
        rating_col = "Star Rating"
        if rating_col not in df.columns:
            if 'rating' in df.columns:
                rating_col = 'rating'
            else:
                # Add a dummy rating if missing
                df[rating_col] = 0
        
        # 3. Review_ID
        if "Review_ID" not in df.columns:
            df["Review_ID"] = range(1, len(df) + 1)
        
        print(f"Running sentiment analysis on {len(df)} reviews...")
        
        # Apply sentiment analysis
        results = df[text_col].apply(get_sentiment_details)
        results_df = pd.DataFrame(results.tolist())
        
        # Combine
        df = pd.concat([df, results_df], axis=1)
        
        # Rename and select columns to match the user's table exactly
        output_df = df.rename(columns={
            text_col: "Review_Text",
            rating_col: "Rating"
        })
        
        columns_to_keep = [
            "Review_ID", "Review_Text", "Rating", 
            "Predicted_Sentiment", "Positive_Prob", 
            "Neutral_Prob", "Negative_Prob", "Tokens", "Token_Vectors"
        ]
        
        # Keep other original columns too but ensure requested ones are first
        existing_cols = [c for c in columns_to_keep if c in output_df.columns]
        other_cols = [c for c in output_df.columns if c not in columns_to_keep]
        output_df = output_df[existing_cols + other_cols]

        print("\nPredicted Sentiment distribution:")
        print(output_df["Predicted_Sentiment"].value_counts())
        
        # Save the result
        output_path = output_dir / file_path.name.replace("_clean.csv", "_sentiment.csv")
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved sentiment results to: {output_path}")

if __name__ == "__main__":
    process_files()
