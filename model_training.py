# model_training.py

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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
