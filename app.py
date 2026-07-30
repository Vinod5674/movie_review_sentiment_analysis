from flask import Flask, render_template, request
import nltk
nltk.download("stopwards")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")
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
    app.run(debug=False)
