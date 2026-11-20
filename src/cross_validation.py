import pandas as pd
import csv

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
# Load the SMS Spam Collection dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    keep_default_na=False,
    quoting=csv.QUOTE_NONE
)

# Remove duplicate records
df_clean = df.drop_duplicates(
    subset=["message", "label"]
).copy()

# Separate messages and labels
X = df_clean["message"]
y = df_clean["label"]

# Reproduce the original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Total cleaned records:", len(df_clean))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Define the machine learning models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

# Configure 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nCross-validation configuration:")
print("Number of folds:", cv.get_n_splits())
print("Number of models:", len(models))

# Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

# Store cross-validation results
cv_results = []

# Evaluate each model
for model_name, model in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", model)
    ])

    print(f"\nEvaluating {model_name}...")
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Accuracy_Mean": scores["test_accuracy"].mean(),
        "Accuracy_Std": scores["test_accuracy"].std(),
        "Precision_Mean": scores["test_precision"].mean(),
        "Recall_Mean": scores["test_recall"].mean(),
        "F1_Mean": scores["test_f1"].mean()
    })

    print("Fold Accuracy:", scores["test_accuracy"])
    print("Mean Accuracy:", scores["test_accuracy"].mean())
    print("Mean F1-Score:", scores["test_f1"].mean())

results_df = pd.DataFrame(cv_results)

print("\n" + "=" * 60)
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# Save cross-validation results
results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\nCross-validation results saved successfully!")
