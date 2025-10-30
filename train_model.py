import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X_train = train_df['Cleaned_Comment']
y_train = train_df['Sentiment']
X_test = test_df['Cleaned_Comment']
y_test = test_df['Sentiment']

# Vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("model/sentiment_model.pkl", "wb"))
print("\n✅ Model and vectorizer saved successfully!")
