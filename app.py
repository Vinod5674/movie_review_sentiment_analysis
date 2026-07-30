from flask import Flask, render_template, request
import nltk
import os

nltk_data_path ="/opt/render/nltk_data"

os.makedirs(nltk_data_path , exist_ok=True)

nltk.data.path.append(nltk_data_path)

resources = [
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet"
]

for resouce in resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource,download_dir = nltk_data_path)
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
    app.run(host = "0.0.0.0" , port=5000)
