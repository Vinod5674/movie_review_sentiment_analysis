from flask import Flask, render_template, request
import nltk
import os

from src.predict import predict_sentiment


# NLTK resources download for Render
nltk_data_path = "/opt/render/nltk_data"

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

resources = [
    "stopwords",
    "punkt",
    "wordnet",
    "omw-1.4"
]

for resource in resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)


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
    app.run(host="0.0.0.0", port=5000)