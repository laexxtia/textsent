import pandas as pd
from preprocessing import preprocess_data
from feature import create_features
from model_training import (
    select_top_features, 
    train_naive_bayes, 
    train_naive_bayes_with_raw_counts, 
    train_logistic_regression_with_tfidf, 
    train_logistic_regression_with_raw_counts
)
from bert_training import preprocess_for_bert, train_bert, evaluate_bert
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time
from transformers import DistilBertTokenizer

def main():
    # Load and preprocess the data
    filepath = 'labeled_data.csv'  # Adjust this path
    preprocessed_data = preprocess_data(filepath, 'tweet')
    
    # Create features
    start_time = time.time()
    features = create_features(preprocessed_data)
    print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")

    # Prepare data for models
    X = features.drop(columns=['compound'])  # Exclude columns not used as features
    y = preprocessed_data['class']  # Make sure this is the correct column name for your labels

    # Feature selection: Select top predictive features
    start_time = time.time()
    X_selected, top_features = select_top_features(X, y, k=10)
    print(f"Feature selection time: {time.time() - start_time:.2f} seconds")
    print("Top predictive features:", top_features)

    # Split the data for traditional models
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train and evaluate Naive Bayes with TF-IDF features
    start_time = time.time()
    nb_model_with_tfidf = train_naive_bayes(X_train, y_train)
    y_pred_with_tfidf = nb_model_with_tfidf.predict(X_test)
    print(f"Naive Bayes with TF-IDF training time: {time.time() - start_time:.2f} seconds")
    print("Classification Report for Naive Bayes with TF-IDF:")
    nb_classification_report = classification_report(y_test, y_pred_with_tfidf, output_dict=True)
    print(classification_report(y_test, y_pred_with_tfidf))
    print(f"Accuracy with TF-IDF: {accuracy_score(y_test, y_pred_with_tfidf):.2%}")

    # Train and evaluate Naive Bayes with raw counts
    X_raw = preprocessed_data['processed_text']  # Use raw text
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    start_time = time.time()
    nb_model_without_tfidf, count_vectorizer_nb = train_naive_bayes_with_raw_counts(X_train_raw, y_train_raw)
    X_test_raw_counts_nb = count_vectorizer_nb.transform(X_test_raw)
    y_pred_without_tfidf = nb_model_without_tfidf.predict(X_test_raw_counts_nb)
    print(f"Naive Bayes without TF-IDF training time: {time.time() - start_time:.2f} seconds")
    print("Classification Report for Naive Bayes without TF-IDF:")
    nb_raw_classification_report = classification_report(y_test_raw, y_pred_without_tfidf, output_dict=True)
    print(classification_report(y_test_raw, y_pred_without_tfidf))
    print(f"Accuracy without TF-IDF: {accuracy_score(y_test_raw, y_pred_without_tfidf):.2%}")

    # Train and evaluate Logistic Regression with TF-IDF features
    start_time = time.time()
    lr_model_with_tfidf = train_logistic_regression_with_tfidf(X_train, y_train)
    y_pred_lr_with_tfidf = lr_model_with_tfidf.predict(X_test)
    print(f"Logistic Regression with TF-IDF training time: {time.time() - start_time:.2f} seconds")
    print("Classification Report for Logistic Regression with TF-IDF:")
    lr_classification_report = classification_report(y_test, y_pred_lr_with_tfidf, output_dict=True)
    print(classification_report(y_test, y_pred_lr_with_tfidf))
    print(f"Accuracy with TF-IDF (Logistic Regression): {accuracy_score(y_test, y_pred_lr_with_tfidf):.2%}")

    # Train and evaluate Logistic Regression with raw counts
    start_time = time.time()
    lr_model_without_tfidf, count_vectorizer_lr = train_logistic_regression_with_raw_counts(X_train_raw, y_train_raw)
    X_test_raw_counts_lr = count_vectorizer_lr.transform(X_test_raw)
    y_pred_lr_without_tfidf = lr_model_without_tfidf.predict(X_test_raw_counts_lr)
    print(f"Logistic Regression without TF-IDF training time: {time.time() - start_time:.2f} seconds")
    print("Classification Report for Logistic Regression without TF-IDF:")
    lr_raw_classification_report = classification_report(y_test_raw, y_pred_lr_without_tfidf, output_dict=True)
    print(classification_report(y_test_raw, y_pred_lr_without_tfidf))
    print(f"Accuracy without TF-IDF (Logistic Regression): {accuracy_score(y_test_raw, y_pred_lr_without_tfidf):.2%}")

    # Train and evaluate BERT model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Use DistilBERT for faster training
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        preprocessed_data['processed_text'], preprocessed_data['class'], test_size=0.2, random_state=42)
    train_dataset = preprocess_for_bert(train_texts, train_labels, tokenizer)
    test_dataset = preprocess_for_bert(test_texts, test_labels, tokenizer)
    start_time = time.time()
    trainer = train_bert(train_dataset, test_dataset)
    print(f"BERT training time: {time.time() - start_time:.2f} seconds")

    # Evaluate BERT model
    start_time = time.time()
    evaluation_results = evaluate_bert(trainer)
    print(f"BERT evaluation time: {time.time() - start_time:.2f} seconds")
    print("BERT Evaluation Results:", evaluation_results)

    # Optional: Convert BERT evaluation results to a similar format as classification_report
    bert_eval_results = {key: evaluation_results[key] for key in ['eval_loss']}
    bert_eval_results['accuracy'] = evaluation_results['eval_accuracy'] if 'eval_accuracy' in evaluation_results else None

    # Print summary comparison
    print("\nSummary Comparison:")
    print(f"Naive Bayes with TF-IDF: Accuracy: {accuracy_score(y_test, y_pred_with_tfidf):.2%}, Loss: {nb_classification_report['macro avg']['f1-score']}")
    print(f"Naive Bayes with Raw Counts: Accuracy: {accuracy_score(y_test_raw, y_pred_without_tfidf):.2%}, Loss: {nb_raw_classification_report['macro avg']['f1-score']}")
    print(f"Logistic Regression with TF-IDF: Accuracy: {accuracy_score(y_test, y_pred_lr_with_tfidf):.2%}, Loss: {lr_classification_report['macro avg']['f1-score']}")
    print(f"Logistic Regression with Raw Counts: Accuracy: {accuracy_score(y_test_raw, y_pred_lr_without_tfidf):.2%}, Loss: {lr_raw_classification_report['macro avg']['f1-score']}")
    print(f"BERT: Accuracy: {bert_eval_results['accuracy']}, Loss: {bert_eval_results['eval_loss']}")

if __name__ == '__main__':
    main()
