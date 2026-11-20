import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
# Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    keep_default_na=False,
    quoting=csv.QUOTE_NONE
)

# Remove duplicates
df_clean = df.drop_duplicates(
    subset=["message", "label"]
).copy()

# Separate features and labels
X = df_clean["message"]
y = df_clean["label"]

# Reproduce the original split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset prepared successfully!")
# Create Linear SVM pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LinearSVC())
])

# Train model
model.fit(X_train, y_train)

# Predict test messages
y_pred = model.predict(X_test)

print("\nLinear SVM evaluation:")
print(classification_report(y_test, y_pred))

# Create a results DataFrame
error_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Message": X_test.values
})

# Identify false positives
false_positives = error_df[
    (error_df["Actual"] == "ham") &
    (error_df["Predicted"] == "spam")
]

# Identify false negatives
false_negatives = error_df[
    (error_df["Actual"] == "spam") &
    (error_df["Predicted"] == "ham")
]

print("\nError Analysis")
print("-" * 50)

print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

print("\nFalse Positive Examples:")
print(false_positives[["Actual", "Predicted", "Message"]].head(5).to_string(index=False))

print("\nFalse Negative Examples:")
print(false_negatives[["Actual", "Predicted", "Message"]].head(5).to_string(index=False))