# https://medium.com/@sebastienwebdev/sentiment-analysis-on-e-commerce-reviews-c3c42edffb13
# adapted to be used with polars and to our needs (simply copy-pasting is not enough)

import polars as pl
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ''  # If text is not a string (e.g., NaN), return an empty string
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in stopwords.words('english')]

#     return ' '.join(tokens)

# Question: Implement a document clustering algorithm using NLP techniques for feature extraction


word_blacklist = ['idk']
pos_blacklist = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRON', 'PRT', 'VERB', 'X']

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def preprocess_text(text: str):
    new_text = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(text)):
        if word in stopwords.words('english'):
            print(f'skipping stopword: {word}')
            continue
        if not word.isalpha():
            print(f'skipping non-alpha: {word}')        
            continue
        if word in word_blacklist:
            print(f'skipping blacklisted word: {word}')
            continue
        if pos in pos_blacklist:
            print(f'skipping blacklisted pos: {pos}')
            continue
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        if pos in pos_blacklist:
            print(f'skipping blacklisted pos: {pos}')
            continue
        new_text.append(lemmatized_word)        # choose a good stemmer --- nltk.PorterStemmer().stem(word)
    return ' '.join(new_text)

def preprocess_data(data: list[str]):
    return [preprocess_text(text) for text in data]

def count_vectorize(data):
    print("Data")
    print(data)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.7, min_df=0.2)
    X = vectorizer.fit_transform(data)
    print("Features")
    print(vectorizer.get_feature_names_out())
    print("--------------------------------")
    print(X.toarray())


#### Patrick's code for sentiment analysis
# def run(data, filename):
#     df = pl.DataFrame({"reviews": data})
    
#     df = df.with_columns(  
#         pl.col("reviews")
#         .map_elements(preprocess_text)
#         .alias("cleaned_reviews"),
#     )
    
#     sia = SentimentIntensityAnalyzer()
#     df = df.with_columns(  
#         pl.col("cleaned_reviews")
#         .map_elements(lambda x: sia.polarity_scores(x))
#         .alias("sentiments"),
#     )
#     df = df.with_columns(  
#         pl.col("sentiments")
#         .map_elements(lambda score_dict: score_dict['compound'])
#         .alias("compound"),
#     )
#     df = df.with_columns(  
#         pl.col("compound")
#         .map_elements(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
#         .alias("sentiment_type"),
#     )

#     df = df.drop(["compound", "sentiments", "cleaned_reviews"])

#     df.write_json(filename)
#     print(f"Processed data saved to {filename}")