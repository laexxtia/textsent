import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
import emoji

# Initialize the tokenizer
tokenizer = TweetTokenizer()

def preprocess_tweet(text):
    # Convert emojis to words
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Handling common emoticons by converting them to a word (optional, you can add more)
    emoticons = {':)': 'smile', ':(': 'sad', ':D': 'laugh'}
    for emoticon, word in emoticons.items():
        text = text.replace(emoticon, f' {word} ')
    # Tokenize and convert to lower case
    tokens = tokenizer.tokenize(text.lower())
    # Further clean tokens by removing any remaining special characters
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]
    # Return a single string of tokens
    return ' '.join(tokens)

def preprocess_data(filepath, text_column):
    # Load data
    data = pd.read_csv(filepath)
    # Apply preprocessing to each tweet
    data['processed_text'] = data[text_column].apply(preprocess_tweet)
    return data