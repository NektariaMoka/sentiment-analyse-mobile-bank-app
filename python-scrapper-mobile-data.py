from google_play_scraper import reviews
import pandas as pd

result, continuation_token = reviews(
    'gr.alpha.alpha_mobile_banking',
    lang='en',
    country='gr',
    count=5000
)

df = pd.DataFrame(result)

print(df.columns)
print(df.head())