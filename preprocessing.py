import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    return text

def tokenize_and_normalize(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_data(filepath):
    data = load_data(filepath)
    data['clean_text'] = data['tweet'].apply(clean_text)
    data['tokenized_text'] = data['clean_text'].apply(tokenize_and_normalize)
    return data
