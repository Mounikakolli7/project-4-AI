import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1️⃣ Load dataset
df = pd.read_csv("dataset.csv")

# 2️⃣ Features and labels
X = df["text"]
y = df["label"]

# 3️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 5️⃣ Train a classifier (Naive Bayes is common for text)
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 6️⃣ Evaluate the model
y_pred = model.predict(X_test_vect)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save the trained model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
