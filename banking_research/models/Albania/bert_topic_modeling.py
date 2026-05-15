import pandas as pd
import numpy as np
import os
from bertopic import BERTopic
from pathlib import Path

# Setup paths relative to the script
base_dir = Path(__file__).parent
data_dir = base_dir / "data/cleaned"
output_dir = base_dir / "data/topics"
output_dir.mkdir(parents=True, exist_ok=True)

def process_files():
    # Find all cleaned CSV files for Albania
    files = list(data_dir.glob("albania_*_clean.csv"))
    
    if not files:
        print(f"No cleaned files found in {data_dir}")
        return

    for file_path in files:
        print(f"\nProcessing {file_path.name} for topic modeling...")
        df = pd.read_csv(file_path)
        
        # Identify the text column (using same logic as sentiment_analysis.py)
        text_col = "Review Text"
        if text_col not in df.columns:
            if 'review' in df.columns:
                text_col = 'review'
            elif 'clean_review' in df.columns:
                text_col = 'clean_review'
            else:
                print(f"Could not find text column in {file_path.name}. Skipping...")
                continue

        # Identify the rating column
        rating_col = "Star Rating"
        if rating_col not in df.columns:
            if 'rating' in df.columns:
                rating_col = 'rating'
            else:
                rating_col = None
        
        # Ensure we have clean strings
        docs = df[text_col].astype(str).tolist()

        print(f"Training BERTopic model on {len(docs)} reviews...")

        topic_model = BERTopic(
            language="multilingual",
            calculate_probabilities=True,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(docs)
        df["topic"] = topics

        # Generate dynamic labels from the model itself
        # This takes the top 3 words and joins them with an underscore
        topic_labels = topic_model.generate_topic_labels(nr_words=3, separator="_")
        # Set the custom labels in the model
        topic_model.set_topic_labels(topic_labels)
        
        # Add labels to the dataframe using the model's generated labels
        # get_document_info(docs) returns all info per document including our new CustomName
        doc_info = topic_model.get_document_info(docs)
        df["topic_label"] = doc_info["CustomName"].values
        
        # Extract the top keywords as a list
        # doc_info["Representation"] contains a list of words for each doc's topic
        df["top_keywords"] = doc_info["Representation"]

        print("\nTopic overview:")
        print(topic_model.get_topic_info())

        # ==============================
        # TOPIC PROBABILITIES
        # ==============================
        # probs is a numpy array of shape (n_docs, n_topics)
        # Round to 3 decimal places as requested
        probs_rounded = np.round(probs, 3)
        topic_prob_df = pd.DataFrame(probs_rounded)
        
        # Get labels for all topics (excluding -1 if it's not in probs)
        # BERTopic probabilities usually correspond to topics 0, 1, 2...
        # We'll use the CustomName we set earlier for these columns
        topic_info = topic_model.get_topic_info()
        # Filter out outlier topic -1 for mapping labels to probability columns
        actual_topics = topic_info[topic_info["Topic"] != -1].copy()
        # The probability matrix 'probs' has columns for topics 0, 1, 2... in order
        # So we sort actual_topics by Topic ID to match column order
        actual_topics = actual_topics.sort_values("Topic")
        new_column_names = actual_topics["CustomName"].tolist()
        
        # In case the number of topics doesn't match columns for some reason (rare)
        if len(new_column_names) == topic_prob_df.shape[1]:
            topic_prob_df.columns = new_column_names
        else:
            topic_prob_df.columns = [
                f"topic_{i}" for i in range(topic_prob_df.shape[1])
            ]
            
        topic_prob_df = topic_prob_df.replace([np.inf, -np.inf], np.nan)
        topic_prob_df = topic_prob_df.fillna(0)

        # Combine with original dataframe
        df_result = pd.concat(
            [df.reset_index(drop=True), topic_prob_df],
            axis=1
        )

        # Filter to keep only specific columns as requested
        # Review Text, Star Rating, topic, topic_label, top_keywords + all probability columns
        prob_cols = topic_prob_df.columns.tolist()
        cols_to_keep = [text_col, rating_col, "topic", "topic_label", "top_keywords"] + prob_cols
        # Filter existing columns only to avoid errors if rating is missing
        final_cols = [c for c in cols_to_keep if c in df_result.columns]
        df_final = df_result[final_cols]

        # Save the result
        output_path = output_dir / file_path.name.replace("_clean.csv", "_topics.csv")
        df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved filtered topic modeling results to: {output_path}")

if __name__ == "__main__":
    process_files()
