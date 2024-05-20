import re
from nltk.tokenize import TweetTokenizer
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# Initialize the tokenizer
tokenizer = TweetTokenizer()

# Helper function to convert emoticons to words
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = text.replace(emot, " " + EMOTICONS[emot] + " ")
    return text

# Helper function to convert emojis to words
def convert_emojis(text):
    for emo in UNICODE_EMO:
        text = text.replace(emo, " " + UNICODE_EMO[emo].replace(",", "").replace(":", "").replace("_", " ") + " ")
    return text

def preprocess_tweet(text):
    # Convert emojis to words
    text = convert_emojis(text)
    # Convert emoticons to words
    text = convert_emoticons(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Tokenize and convert to lower case
    tokens = tokenizer.tokenize(text.lower())
    # Normalize text by removing any remaining special characters
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]
    return tokens
