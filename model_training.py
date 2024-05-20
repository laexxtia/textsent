from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from preprocessing import preprocess_tweet

def train_naive_bayes(X_train, y_train):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report

def setup_bert_data(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_data = data.apply(lambda x: tokenizer(x['tokenized_text'], padding="max_length", truncation=True), axis=1)
    model_data['labels'] = data['class']
    return model_data

def train_evaluate_bert(data):
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    return trainer

def main():
    filepath = 'labeled_data.csv'
    data = preprocess_tweet(filepath)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['tokenized_text'])
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb_model = train_naive_bayes(X_train, y_train)
    print("Naive Bayes Model Evaluation:\n", evaluate_model(nb_model, X_test, y_test))

    bert_data = setup_bert_data(data)
    bert_trainer = train_evaluate_bert(bert_data)
    print("BERT Model Evaluation:\n", bert_trainer.evaluate())

if __name__ == '__main__':
    main()
