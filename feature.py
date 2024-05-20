import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

def create_features(data):
    # Ensure the Vader lexicon is downloaded
    nltk.download('vader_lexicon')
    
    # Initialize TF-IDF Vectorizer with n-grams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=10000)
    
    # Apply TF-IDF to the processed text data
    tfidf_features = tfidf_vectorizer.fit_transform(data['processed_text'].astype('U'))  # Ensure data is unicode

    # Initialize NLTK's VADER Sentiment Intensity Analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores
    data['sentiments'] = data['processed_text'].apply(sid.polarity_scores)
    data['compound'] = data['sentiments'].apply(lambda score_dict: score_dict['compound'])

    # Convert sparse matrix to DataFrame and concatenate with sentiment scores
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    feature_df = pd.concat([tfidf_df, data[['compound']]], axis=1)
    
    return feature_df

def main():
    # Load and preprocess the data
    filepath = 'labeled_data.csv'
    preprocessed_data = preprocess_data(filepath, 'tweet_text')  # Adjust this if the column name is different
    
    # Create features
    features = create_features(preprocessed_data)
    print(features.head())  # Display some of the features to verify

if __name__ == '__main__':
    main()
