# main.py

# Import necessary libraries
from sklearn.model_selection import train_test_split
from preprocessing import load_data, clean_text, tokenize_and_normalize, vectorize_text

def main():
    # Path to the dataset file
    file_path = 'labeled_data.csv'
    
    # Load and preprocess the data
    data = load_data(file_path)
    data['clean_text'] = data['tweet'].apply(clean_text)
    data['tokenized_text'] = data['clean_text'].apply(tokenize_and_normalize)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['tokenized_text'], data['class'], test_size=0.2, random_state=42)
    
    # Vectorize the text data
    X_train_tfidf, vectorizer = vectorize_text(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Now, X_train_tfidf and X_test_tfidf are ready to be used with machine learning models.
    print("Data preprocessing complete. Train and test sets are ready for model training.")

    print(data)

if __name__ == '__main__':
    main()
