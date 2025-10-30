from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and vectorizer
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("model/sentiment_model.pkl", "rb"))

# Create FastAPI app
app = FastAPI(
    title="Tulu-English Sentiment Analyzer",
    description="Predicts sentiment using TF-IDF + Logistic Regression",
    version="1.0"
)

# ----- Enable CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify frontend origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class CommentInput(BaseModel):
    comment: str

# Root route
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API (TF-IDF + Logistic Regression)"}

# Predict route
@app.post("/predict")
def predict_sentiment(data: CommentInput):
    comment = data.comment
    vector = tfidf.transform([comment])

    # Prediction and confidence
    probs = model.predict_proba(vector)[0]
    prediction = model.classes_[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    return {
        "input_comment": comment,
        "predicted_sentiment": prediction,
        "confidence (%)": round(confidence, 2)
    }
