import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_raiffeisen_histogram():
    file_path = Path("../../../data/cleaned/albania_raiffeisen_on_clean.csv")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Columns in dataframe:", df.columns.tolist())
    
    # Based on run_albania_only.py, the column should be 'score'
    rating_col = 'score' if 'score' in df.columns else 'rating'
    
    if rating_col not in df.columns:
        print(f"Error: Column '{rating_col}' not found.")
        return

    print(f"\nValue counts for Raiffeisen Bank of Albania ({len(df)} reviews):")
    print(df[rating_col].value_counts().sort_index())

    # Set the style
    sns.set_theme(style="whitegrid")
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=rating_col, data=df, palette="viridis", hue=rating_col, legend=False)
    
    plt.title("Distribution of User Ratings - Raiffeisen Bank Albania", fontsize=15)
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add counts on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    output_plot = "raiffeisen_albania_histogram.png"
    plt.savefig(output_plot)
    print(f"\nHistogram saved to {output_plot}")

if __name__ == "__main__":
    generate_raiffeisen_histogram()
