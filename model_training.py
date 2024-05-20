# model_training.py

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def select_top_features(X, y, k=10):
    """Select top k predictive features using SelectKBest."""
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    top_features = X.columns[selector.get_support(indices=True)]
    return X_selected, top_features

def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes model using TF-IDF features."""
    nb_pipeline = Pipeline([
        ('clf', MultinomialNB()),
    ])
    nb_pipeline.fit(X_train, y_train)
    return nb_pipeline

def train_naive_bayes_with_raw_counts(X_train, y_train):
    """Train a Naive Bayes model using raw text counts."""
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_counts, y_train)
    return nb_model, count_vectorizer

def train_logistic_regression_with_tfidf(X_train, y_train):
    """Train a Logistic Regression model using TF-IDF features."""
    lr_pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),  # Transform counts to TF-IDF
        ('clf', LogisticRegression(max_iter=1000)),  # Logistic Regression classifier
    ])
    lr_pipeline.fit(X_train, y_train)
    return lr_pipeline

def train_logistic_regression_with_raw_counts(X_train, y_train):
    """Train a Logistic Regression model using raw text counts."""
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_counts, y_train)
    return lr_model, count_vectorizer
