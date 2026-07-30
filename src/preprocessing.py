import re
import string
import nltk

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Download NLTK resources if missing
# -------------------------

resources = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4")
]

for path, name in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

# -------------------------
# Initialize
# -------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_url(text):
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub("", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):

    text = text.lower()

    text = remove_html(text)

    text = remove_url(text)

    text = remove_punctuation(text)

    tokens = word_tokenize(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_words(tokens)

    return " ".join(tokens)