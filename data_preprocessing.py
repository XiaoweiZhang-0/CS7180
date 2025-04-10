import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# a function to tokenize, normalize and remove stop words
def preprocess_text(text):
    # Initialize the stemmer
    stemmer = PorterStemmer()

    text = remove_hashtags_and_mentions(text)

    # Tokenization and normalization
    tokens = text.lower().split()

    # Remove punctuation
    tokens = [word.strip(".,!?;:()[]{}") for word in tokens]

    # Stem or lemmatize the tokens
    tokens = [stemmer.stem(word) for word in tokens]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


def remove_hashtags_and_mentions(text):
    # Remove hashtags and mentions
    text = re.sub(r"#[\w-]+", "", text)  # Remove hashtags
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    return text
