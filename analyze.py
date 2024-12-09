# https://medium.com/@sebastienwebdev/sentiment-analysis-on-e-commerce-reviews-c3c42edffb13
# adapted to be used with polars and to our needs (simply copy-pasting is not enough)

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

import openai
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

openai.api_key = ""

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('vader_lexicon', force=True)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''  # If text is not a string (e.g., NaN), return an empty string
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

def get_clusters(reviews):
    for review in reviews:
        review = preprocess_text(review)
        
    embeddings = [get_embedding(review) for review in reviews]
    X = np.array(embeddings)
    
    silhouette_scores = []
    for k in range(2, len(reviews)-1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    best_k = 2 + silhouette_scores.index(max(silhouette_scores))

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X)

    df = pd.DataFrame({"Review": reviews, "Cluster": labels})
    clusters = df.groupby("Cluster")["Review"].apply(list)
    return clusters

def extract_keywords(reviews_in_cluster, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(reviews_in_cluster)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    
   
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return [keyword for keyword, _ in keywords[:top_n]]    

def get_sentiment(review):
    cleaned_review = preprocess_text(review)
    sia = SentimentIntensityAnalyzer()
    compound = sia.polarity_scores(cleaned_review)["compound"]
    return 'positive' if compound > 0 else ('negative' if compound < 0 else 'neutral')

def analyse(reviews):
    help_reviews = []
    for review in reviews: 
        help_reviews.append(review["content"][:-10]) # remove " Read more"
    reviews = help_reviews
    result = []
    clusters = get_clusters(reviews)
    
    for cluster_id, reviews_in_cluster in clusters.items():
        keywords = extract_keywords(reviews_in_cluster)
        sub_reviews = []
        for review in reviews_in_cluster[:]:  
            sub_reviews.append({
                'review': review,
                'sentiment': get_sentiment(review)
            })
        result.append({
            'cluster': cluster_id,
            'reviews': sub_reviews,
            'keywords': keywords
        })
    return result
