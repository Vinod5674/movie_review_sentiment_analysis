import pickle

from src.preprocessing import preprocess_text


# Load Logistic Regression Model
with open("models/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF Vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)



def predict_sentiment(review):

    clean_review = preprocess_text(review)

    review_vector = vectorizer.transform([clean_review])

    prediction = model.predict(review_vector)

    probability = model.predict_proba(review_vector)

    confidence = max(probability[0]) * 100

    return prediction[0], round(confidence, 2)