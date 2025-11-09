# train_model.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X_train = train_df['Cleaned_Comment']
y_train = train_df['Sentiment']
X_test = test_df['Cleaned_Comment']
y_test = test_df['Sentiment']

# Vectorizers and algorithms
vectorizers = {
    "CountVectorizer": CountVectorizer(),
    "TFIDF": TfidfVectorizer()
}
algorithms = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

results = []
os.makedirs("model", exist_ok=True)

# Loop through all 4 combinations
for vec_name, vec in vectorizers.items():
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    for algo_name, model in algorithms.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted') * 100
        rec = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        # Save model and vectorizer
        pickle.dump(vec, open(f"model/{vec_name}_{algo_name}_vectorizer.pkl", "wb"))
        pickle.dump(model, open(f"model/{vec_name}_{algo_name}_model.pkl", "wb"))

        results.append({
            "Vectorizer": vec_name,
            "Algorithm": algo_name,
            "Accuracy (%)": round(acc, 2),
            "Precision (%)": round(prec, 2),
            "Recall (%)": round(rec, 2),
            "F1-score (%)": round(f1, 2)
        })

# Save results as CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model/model_results.csv", index=False)

print("\n✅ All models trained and saved successfully!")
print(results_df)
