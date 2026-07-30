from flask import Flask, render_template, request
import nltk
import os


# -----------------------------
# NLTK Setup for Render
# -----------------------------

nltk_data_path = "/opt/render/nltk_data"

os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

resources = [
    ("corpora/stopwords","stopwords"),
    ("tokenizers/punkt","punkt"),
    ("corpora/wordnet","wordnet"),
    ("corpora/omw-1.4","omw-1.4"),
]


for path, package in resources:
    try:
        nltk.data.find(path)

    except LookupError:
        nltk.download(
            package,
            download_dir=nltk_data_path
        )


# -----------------------------
# Flask App
# -----------------------------
from src.predict import predict_sentiment


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    result, confidence = predict_sentiment(review)


    if result == "positive":
        prediction = "😊 Positive Review"
    else:
        prediction = "😞 Negative Review"


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )