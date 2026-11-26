
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parents[1]

train_df = pd.read_csv(BASE_DIR / "data" / "train.csv")
test_df = pd.read_csv(BASE_DIR / "data" / "test.csv")

for df in (train_df, test_df):
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

X_train = train_df[["text", "channel"]]
y_train = train_df["priority"]

X_test = test_df[["text", "channel"]]
y_test = test_df["priority"]

preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "text"),
    ("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"])
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        C=0.1,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)

results = {
    "model": "TF-IDF + Logistic Regression",
    "train_records": len(train_df),
    "test_records": len(test_df),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "classification_report": report
}

(BASE_DIR / "results" / "ml_baseline_results.json").write_text(
    json.dumps(results, indent=4)
)

print("ML Baseline Evaluation")
print("----------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nSaved: results/ml_baseline_results.json")
