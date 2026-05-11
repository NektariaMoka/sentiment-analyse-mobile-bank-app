## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Scrape data:**
   - Use the notebook `notebooks/01_scraping.ipynb`
   - OR run the scraper script directly:
     ```bash
     cd banking_research/src
     python scraper.py
     ```
   - You can modify `apps_config.json` to add or remove countries and apps.

3. **Requirements:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   pip install transformers sentence-transformers bertopic
   pip install google-play-scraper nltk spacy
   ```