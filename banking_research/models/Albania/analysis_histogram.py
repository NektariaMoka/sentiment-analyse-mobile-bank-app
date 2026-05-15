import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_rating_histogram():
    base_dir = Path(__file__).parent
    # Use the combined Greece dataset as it has enough data for 1000 reviews
    # or combined Albania if available.
    file_path = base_dir / "data/cleaned/albania_raiffeisen_on_clean.csv"
    
    if not file_path.exists():
        # Try relative to script location if current dir is different
        file_path = Path(__file__).parent / "data/cleaned/albania_raiffeisen_on_clean.csv"

    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for the correct column name in the cleaned Albania dataset
    print("Columns in dataframe:", df.columns.tolist())
    
    # Priority for column names based on the scraper's output format
    if 'Star Rating' in df.columns:
        rating_col = 'Star Rating'
    elif 'score' in df.columns:
        rating_col = 'score'
    elif 'rating' in df.columns:
        rating_col = 'rating'
    else:
        print(f"Error: Rating column not found in {df.columns.tolist()}")
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

    output_plot = base_dir / "rating_distribution_histogram.png"
    plt.savefig(output_plot)
    print(f"\nHistogram saved to {output_plot}")
    # plt.show() # Commented out to avoid interactive block in non-interactive environment

if __name__ == "__main__":
    generate_rating_histogram()
