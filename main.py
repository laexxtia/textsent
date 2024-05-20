# main.py

import pandas as pd
from preprocessing import preprocess_data
from feature import create_features  # Import the function from features.py
from model_training import (
    select_top_features, 
    train_naive_bayes, 
    train_naive_bayes_with_raw_counts, 
    train_logistic_regression_with_tfidf, 
    train_logistic_regression_with_raw_counts
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load and preprocess the data
    filepath = 'labeled_data.csv'  # Adjust this path
    preprocessed_data = preprocess_data(filepath, 'tweet')
    
    # Create features
    features = create_features(preprocessed_data)
    
    # Prepare data for models
    X = features.drop(columns=['compound'])  # Exclude columns not used as features
    y = preprocessed_data['class']  # Make sure this is the correct column name for your labels

    # Feature selection: Select top predictive features
    X_selected, top_features = select_top_features(X, y, k=10)
    print("Top predictive features:", top_features)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes with TF-IDF features
    nb_model_with_tfidf = train_naive_bayes(X_train, y_train)
    
    # Evaluate the Naive Bayes model with TF-IDF features
    y_pred_with_tfidf = nb_model_with_tfidf.predict(X_test)
    print("Classification Report for Naive Bayes with TF-IDF:")
    print(classification_report(y_test, y_pred_with_tfidf))
    print(f"Accuracy with TF-IDF: {accuracy_score(y_test, y_pred_with_tfidf):.2%}")

    # Train Naive Bayes with raw counts
    X_raw = preprocessed_data['processed_text']  # Use raw text
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    nb_model_without_tfidf, count_vectorizer_nb = train_naive_bayes_with_raw_counts(X_train_raw, y_train_raw)
    
    # Transform the test data using the count vectorizer
    X_test_raw_counts_nb = count_vectorizer_nb.transform(X_test_raw)

    # Evaluate the Naive Bayes model without TF-IDF
    y_pred_without_tfidf = nb_model_without_tfidf.predict(X_test_raw_counts_nb)
    print("Classification Report for Naive Bayes without TF-IDF:")
    print(classification_report(y_test_raw, y_pred_without_tfidf))
    print(f"Accuracy without TF-IDF: {accuracy_score(y_test_raw, y_pred_without_tfidf):.2%}")

    # Train Logistic Regression with TF-IDF features
    lr_model_with_tfidf = train_logistic_regression_with_tfidf(X_train, y_train)
    
    # Evaluate the Logistic Regression model with TF-IDF features
    y_pred_lr_with_tfidf = lr_model_with_tfidf.predict(X_test)
    print("Classification Report for Logistic Regression with TF-IDF:")
    print(classification_report(y_test, y_pred_lr_with_tfidf))
    print(f"Accuracy with TF-IDF (Logistic Regression): {accuracy_score(y_test, y_pred_lr_with_tfidf):.2%}")

    # Train Logistic Regression with raw counts
    lr_model_without_tfidf, count_vectorizer_lr = train_logistic_regression_with_raw_counts(X_train_raw, y_train_raw)
    
    # Transform the test data using the count vectorizer
    X_test_raw_counts_lr = count_vectorizer_lr.transform(X_test_raw)

    # Evaluate the Logistic Regression model without TF-IDF
    y_pred_lr_without_tfidf = lr_model_without_tfidf.predict(X_test_raw_counts_lr)
    print("Classification Report for Logistic Regression without TF-IDF:")
    print(classification_report(y_test_raw, y_pred_lr_without_tfidf))
    print(f"Accuracy without TF-IDF (Logistic Regression): {accuracy_score(y_test_raw, y_pred_lr_without_tfidf):.2%}")

if __name__ == '__main__':
    main()
