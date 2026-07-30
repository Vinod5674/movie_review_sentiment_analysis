import re
import string
import nltk


from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Uncomment these lines only the first time
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub('', text)


def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = remove_html(text)

    # Remove URL
    text = remove_url(text)

    # Remove Punctuation
    text = remove_punctuation(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove Stopwords
    tokens = remove_stopwords(tokens)

    # Lemmatization
    tokens = lemmatize_words(tokens)

    # Convert list into string
    text = " ".join(tokens)

    return text