# 💬 Tulu-English Sentiment Analysis (FastAPI + TF-IDF + Logistic Regression)

This mini-project performs **Sentiment Analysis** on **Tulu-English code-mixed comments**, identifying whether a comment expresses a **Positive**, **Negative**, or **Neutral** sentiment.  
It uses a **TF-IDF Vectorizer** with a **Logistic Regression** classifier trained on custom-labeled data and exposes a **REST API** built with **FastAPI** for real-time predictions.

---

## 🚀 Tech Stack

- **Python 3.12.2**  
- **FastAPI** – for serving the API  
- **Scikit-learn** – for training and evaluating the ML model  
- **NumPy & Pandas** – for data handling and preprocessing  
- **Pickle** – for saving and loading the model/vectorizer

---

## 📂 Project Structure

```
tulu-sentiment-analysis/
│
├── model/
│   ├── CountVectorizer_LogisticRegression_model.pkl
│   ├── CountVectorizer_LogisticRegression_vectorizer.pkl
│   ├── CountVectorizer_NaiveBayes_model.pkl
│   ├── CountVectorizer_NaiveBayes_vectorizer.pkl
│   ├── TFIDF_LogisticRegression_model.pkl
│   ├── TFIDF_LogisticRegression_vectorizer.pkl
│   ├── TFIDF_NaiveBayes_model.pkl
│   ├── TFIDF_NaiveBayes_vectorizer.pkl
│   └── model_results.csv
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── app.py                # FastAPI application
├── train_model.py        # Script to train and save model
├── requirements.txt      # Dependencies
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/me-lw-in/Tulu-English-Sentiment-Analysis.git
cd tulu-sentiment-analysis
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server
```bash
uvicorn app:app --reload
```

### 5. Test in Postman

Visit:  
👉 **[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)**  

You can enter a sample input like:
```json
{
  "comment": "song matra laik ittend"
}
```

Response example:
```json
{
    "input_comment": "song matra laik ittend",
    "predicted_sentiment": "positive",
    "confidence (%)": 83.94
}
```

---

## 🧠 Model Details

- **Vectorizer:** TF-IDF (Term Frequency – Inverse Document Frequency)  
- **Algorithm:** Logistic Regression  
- **Accuracy Achieved:** ~89.86% on test dataset  
- **Dataset:** Custom Tulu-English code-mixed comments (train/test split)  

---

## 🌐 Frontend Integration (Optional)

You can connect this API to a **React** or **any web frontend**.  
Just send a POST request to:

```
http://127.0.0.1:8000/predict
```

with JSON data like:
```json
{
  "comment": "song matra laik ittend"
}
```

---

## 🧾 License

This project was developed as part of an **MCA Mini Project** for academic purposes.  
You are free to use or extend it for learning and research.

---

## 👥 Team Members & Guide

- **Student:** Melwin Manish Mendonca  
- **Guide:** Arhath Kumar

---
