import re
import string

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_url(text):
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub("", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = remove_html(text)

    # Remove URL
    text = remove_url(text)

    # Remove punctuation
    text = remove_punctuation(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


