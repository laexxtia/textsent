# preprocessing.py

# Import necessary libraries for preprocessing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Clean the text data
def clean_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Lowercase all texts
    text = text.lower()
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    return text

# Tokenize and normalize text data
def tokenize_and_normalize(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Vectorize text data using TF-IDF
def vectorize_text(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer

def preprocess_data(filepath):
    data = load_data(filepath)
    data['clean_text'] = data['tweet'].apply(clean_text)
    data['tokenized_text'] = data['clean_text'].apply(tokenize_and_normalize)
    return data