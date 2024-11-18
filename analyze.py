# https://medium.com/@sebastienwebdev/sentiment-analysis-on-e-commerce-reviews-c3c42edffb13
# adapted to be used with polars and to our needs (simply copy-pasting is not enough)

import polars as pl
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def preprocess_text(text):
    if not isinstance(text, str):
        return ''  # If text is not a string (e.g., NaN), return an empty string
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def run(data, filename):
    df = pl.DataFrame({"reviews": data})
    
    df = df.with_columns(  
        pl.col("reviews")
        .map_elements(preprocess_text)
        .alias("cleaned_reviews"),
    )
    
    sia = SentimentIntensityAnalyzer()
    df = df.with_columns(  
        pl.col("cleaned_reviews")
        .map_elements(lambda x: sia.polarity_scores(x))
        .alias("sentiments"),
    )
    df = df.with_columns(  
        pl.col("sentiments")
        .map_elements(lambda score_dict: score_dict['compound'])
        .alias("compound"),
    )
    df = df.with_columns(  
        pl.col("compound")
        .map_elements(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
        .alias("sentiment_type"),
    )

    df = df.drop(["compound", "sentiments", "cleaned_reviews"])

    df.write_json(filename)
    print(f"Processed data saved to {filename}")