import pandas as pd
import numpy as np
import re
import joblib
import shutil
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("==> Cleaning up temporary folders")
shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)

print("==> Loading and preprocessing dataset")
df = pd.read_csv("AI_Human.csv")[["text", "generated"]]
df["label"] = df["generated"].astype(int)

print("==> Cleaning text")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special chars except spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("==> Balancing dataset (optional)")
human_count = df[df.label == 0].shape[0]
ai_count = df[df.label == 1].shape[0]
min_count = min(human_count, ai_count)
df_human_sampled = df[df.label == 0].sample(min_count, random_state=42)
df_ai_sampled = df[df.label == 1].sample(min_count, random_state=42)
df_balanced = pd.concat([df_human_sampled, df_ai_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

print("==> Performing train-test split")
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["clean_text"], df_balanced["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_balanced["label"]
)

print("==> Vectorizing with TF-IDF")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=5,
    max_df=0.8,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("==> Training Logistic Regression with GridSearchCV")
lr = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"]
}
grid = GridSearchCV(lr, params, cv=5, scoring='f1', n_jobs=1, verbose=1)
grid.fit(X_train_tfidf, y_train)

print("==> GridSearch Complete")
print("Best params:", grid.best_params_)
print("Best CV F1 Score:", grid.best_score_)

best_model = grid.best_estimator_

print("==> Evaluating on test set")
y_pred = best_model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
disp.plot()
plt.show()

print("==> Saving model and vectorizer")
joblib.dump(best_model, "logreg_ai_human_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")

print("==> Defining prediction function")
def predict_text(text):
    clean = clean_text(text)
    vect = tfidf.transform([clean])
    pred = best_model.predict(vect)[0]
    prob = best_model.predict_proba(vect)[0]
    return pred, prob

if __name__ == "__main__":
    print("==> Running example prediction")
    example_text = "This is an example text to check if it is AI or human written."
    label, probabilities = predict_text(example_text)
    label_name = "AI" if label == 1 else "Human"
    print(f"Prediction: {label_name}, Confidence: {probabilities}")