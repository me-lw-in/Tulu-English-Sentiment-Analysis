# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, numpy as np, pandas as pd, time, os

# ----- Load Models -----
def load_models():
    models = []
    for file in os.listdir("model"):
        if file.endswith("_model.pkl"):
            base_name = file.replace("_model.pkl", "")
            vectorizer_path = f"model/{base_name}_vectorizer.pkl"

            with open(f"model/{file}", "rb") as m, open(vectorizer_path, "rb") as v:
                model = pickle.load(m)
                vectorizer = pickle.load(v)

            vec_name, algo_name = base_name.split("_")
            models.append({
                "vectorizer_name": vec_name,
                "algorithm_name": algo_name,
                "model": model,
                "vectorizer": vectorizer
            })
    return models

models = load_models()
results_df = pd.read_csv("model/model_results.csv")

# ----- FastAPI app -----
app = FastAPI(title="Tulu-English Sentiment Analyzer", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class CommentInput(BaseModel):
    comment: str


@app.get("/")
def home():
    return {"message": "Welcome to Tulu-English Sentiment Analysis API"}


# ✅ Main model (TF-IDF + Logistic Regression)
@app.post("/predict")
def predict_sentiment(data: CommentInput):
    comment = data.comment
    # Find main model
    main_model = next(
        (m for m in models if m["vectorizer_name"] == "TFIDF" and m["algorithm_name"] == "LogisticRegression"), None
    )
    vector = main_model["vectorizer"].transform([comment])
    probs = main_model["model"].predict_proba(vector)[0]
    prediction = main_model["model"].classes_[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)
    return {
        "input_comment": comment,
        "predicted_sentiment": prediction,
        "confidence (%)": round(confidence, 2)
    }


# 🧠 Compare all models
@app.post("/compare_models")
def compare_models(data: CommentInput):
    comment = data.comment
    all_results = []

    for m in models:
        start = time.time()
        vec = m["vectorizer"].transform([comment])
        probs = m["model"].predict_proba(vec)[0]
        pred = m["model"].classes_[np.argmax(probs)]
        conf = float(np.max(probs) * 100)
        end = time.time()

        # fetch metrics from saved CSV
        row = results_df[
            (results_df["Vectorizer"] == m["vectorizer_name"]) &
            (results_df["Algorithm"] == m["algorithm_name"])
        ].iloc[0]

        explanation = ""
        if m["algorithm_name"] == "NaiveBayes":
            explanation = "Uses word frequency probabilities; simple and fast."
        else:
            explanation = "Learns weighted importance of words; generalizes better."

        all_results.append({
            "vectorizer": m["vectorizer_name"],
            "algorithm": m["algorithm_name"],
            "prediction": pred,
            "confidence (%)": round(conf, 2),
            "accuracy (%)": row["Accuracy (%)"],
            "precision (%)": row["Precision (%)"],
            "recall (%)": row["Recall (%)"],
            "f1-score (%)": row["F1-score (%)"],
            "response_time (s)": round(end - start, 3),
            "explanation": explanation
        })

    # find best model
    best = max(all_results, key=lambda x: x["accuracy (%)"])

    return {
        "input_comment": comment,
        "models": all_results,
        "best_model": f"{best['vectorizer']} + {best['algorithm']}"
    }
