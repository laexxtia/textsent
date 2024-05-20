# main.py

import pandas as pd
from preprocessing import preprocess_data
from feature import create_features  # Import the function from features.py
from model_training import select_top_features, train_naive_bayes
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
    
    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Evaluate the model
    y_pred = nb_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

if __name__ == '__main__':
    main()
