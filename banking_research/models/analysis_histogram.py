import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_rating_histogram():
    # Use the combined Greece dataset as it has enough data for 1000 reviews
    # or combined Albania if available.
    file_path = Path("../data/cleaned/greece_banking_reviews_clean.csv")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # In the previous session, we renamed 'rating' to 'score' for final output
    # or kept it as 'rating' in clean_reviews. Let's check columns.
    print("Columns in dataframe:", df.columns.tolist())
    
    rating_col = 'score' if 'score' in df.columns else 'rating'
    
    if rating_col not in df.columns:
        print(f"Error: Column '{rating_col}' not found.")
        return

    # Take first 1000 reviews as requested
    df_subset = df.head(1000)
    
    print(f"\nValue counts for the first {len(df_subset)} reviews:")
    print(df_subset[rating_col].value_counts().sort_index())

    # Set the style
    sns.set_theme(style="whitegrid")
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=rating_col, data=df_subset, palette="viridis")
    
    plt.title(f"Distribution of User Ratings (Top {len(df_subset)} reviews)", fontsize=15)
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    output_plot = "rating_distribution_histogram.png"
    plt.savefig(output_plot)
    print(f"\nHistogram saved to {output_plot}")
    # plt.show() # Commented out to avoid interactive block in non-interactive environment

if __name__ == "__main__":
    generate_rating_histogram()
