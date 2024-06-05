# textsent
Text Mining &amp; Sentiment Analysis AY23/24 Project

# Hate Speech Detection on Twitter

This project investigates the distinction between hate speech and casual offensive remarks on Twitter by leveraging natural language processing and machine learning techniques. The aim is to develop models that can accurately classify tweets into hate speech, offensive language, and neither.

## Project Structure

.
├── pycache
├── README.md
├── bert_training.py
├── feature.py
├── labeled_data.csv
├── main.ipynb
├── main.py
├── model_training.py
├── preprocessing.py


### Files and Directories

- `__pycache__`: Contains compiled Python files.
- `README.md`: This file, providing an overview of the project and instructions on how to use it.
- `bert_training.py`: Contains the implementation for training and evaluating the BERT model.
- `feature.py`: Contains functions for feature extraction, including TF-IDF vectorization and sentiment analysis.
- `labeled_data.csv`: The dataset used for training and evaluation, containing tweets annotated for hate speech, offensive language, and neither.
- `main.ipynb`: A Jupyter Notebook used for certain visualizations and exploratory data analysis.
- `main.py`: The main script to run the entire pipeline, from data preprocessing to model training and evaluation.
- `model_training.py`: Contains implementations for training and evaluating other models such as Naive Bayes and Logistic Regression.
- `preprocessing.py`: Contains functions for data preprocessing, such as tokenization, normalization, and cleaning.

## How to Run

1. Run the main.py file. This will execute the entire pipeline, including data preprocessing, feature extraction, model training and evaluation.
2. The .ipynb file is used for certain visualizations and exploratory data analysis.