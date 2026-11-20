import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time
from sklearn.svm import LinearSVC

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

# Analyze duplicate records
print("\nDuplicate Record Analysis:")

print("Total duplicate rows:", df.duplicated().sum())

print(
    "Unique messages:",
    df["message"].nunique()
)

print(
    "Unique message-label pairs:",
    df.drop_duplicates(
        subset=["message", "label"]
    ).shape[0]
)

# Create a duplicate-free dataset
df_clean = df.drop_duplicates(
    subset=["message", "label"]
).copy()

print("\nDataset after removing exact duplicates:")
print("Original records:", len(df))
print("Cleaned records:", len(df_clean))
print("\nCleaned label distribution:")
print(df_clean["label"].value_counts())

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

# Define input features and target labels
# Define input features and target labels
# Use the duplicate-free dataset
X = df_clean["message"]
y = df_clean["label"]

print("\nInput Features:")
print(X.head())

print("\nTarget Labels:")
print(y.head())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset split information
print("\nDataset Splitting:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining Label Distribution:")
print(y_train.value_counts())

print("\nTesting Label Distribution:")
print(y_test.value_counts())
# Check for duplicate messages across training and testing sets
train_messages = set(X_train)
test_messages = set(X_test)

overlapping_messages = train_messages.intersection(test_messages)

print("\nDuplicate Leakage Check:")
print("Unique messages in training set:", len(train_messages))
print("Unique messages in testing set:", len(test_messages))
print("Messages appearing in both sets:", len(overlapping_messages))

# Check whether identical messages have conflicting labels
conflicting_labels = (
    df.groupby("message")["label"]
    .nunique()
    .gt(1)
    .sum()
)

print("Messages with conflicting labels:", conflicting_labels)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Fit on training data and transform training messages
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing messages
X_test_tfidf = tfidf.transform(X_test)

# Display TF-IDF information
print("\nTF-IDF Vectorization:")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)
print("Number of extracted features:", len(tfidf.get_feature_names_out()))

# Initialize Multinomial Naive Bayes model
nb_model = MultinomialNB()

# Train the model
# Measure Naive Bayes training time
start_time = time.perf_counter()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

training_time_nb = time.perf_counter() - start_time

print("Training Time:", training_time_nb, "seconds")

print("\nMultinomial Naive Bayes Model:")
print("Model training completed successfully!")

# Generate predictions on testing data
# Measure Naive Bayes inference time
start_time = time.perf_counter()

# Generate predictions
y_pred_nb = nb_model.predict(X_test_tfidf)

inference_time_nb = time.perf_counter() - start_time

print("Inference Time:", inference_time_nb, "seconds")

# Display sample predictions
print("\nSample Model Predictions:")

for actual, predicted in zip(y_test[:10], y_pred_nb[:10]):
    print("Actual:", actual, "| Predicted:", predicted)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_nb)

precision = precision_score(
    y_test, y_pred_nb, pos_label="spam"
)

recall = recall_score(
    y_test, y_pred_nb, pos_label="spam"
)

f1 = f1_score(
    y_test, y_pred_nb, pos_label="spam"
)

# Display evaluation results
print("\nNaive Bayes Model Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred_nb,
    labels=["ham", "spam"]
)

# Display confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/naive_bayes_confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# LOGISTIC REGRESSION MODEL
# ==========================================

# Initialize Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

# Measure training time
start_time = time.perf_counter()

# Train the model
lr_model.fit(X_train_tfidf, y_train)

training_time_lr = time.perf_counter() - start_time

print("\nLogistic Regression Model:")
print("Model training completed successfully!")
print("Training Time:", training_time_lr, "seconds")

# Generate predictions
start_time = time.perf_counter()

y_pred_lr = lr_model.predict(X_test_tfidf)

inference_time_lr = time.perf_counter() - start_time

print("Inference Time:", inference_time_lr, "seconds")

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)

precision_lr = precision_score(
    y_test, y_pred_lr, pos_label="spam"
)

recall_lr = recall_score(
    y_test, y_pred_lr, pos_label="spam"
)

f1_lr = f1_score(
    y_test, y_pred_lr, pos_label="spam"
)

print("\nLogistic Regression Evaluation:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-Score:", f1_lr)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion matrix
cm_lr = confusion_matrix(
    y_test,
    y_pred_lr,
    labels=["ham", "spam"]
)

print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)

# Plot confusion matrix
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Ham", "Spam"]
)

disp_lr.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/logistic_regression_confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# LINEAR SVM MODEL
# ==========================================

# Initialize Linear SVM
svm_model = LinearSVC(random_state=42, max_iter=5000)

# Measure training time
start_time = time.perf_counter()

# Train the model
svm_model.fit(X_train_tfidf, y_train)

training_time_svm = time.perf_counter() - start_time

print("\nLinear SVM Model:")
print("Model training completed successfully!")
print("Training Time:", training_time_svm, "seconds")

# Generate predictions
start_time = time.perf_counter()

y_pred_svm = svm_model.predict(X_test_tfidf)

inference_time_svm = time.perf_counter() - start_time

print("Inference Time:", inference_time_svm, "seconds")

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)

precision_svm = precision_score(
    y_test, y_pred_svm, pos_label="spam"
)

recall_svm = recall_score(
    y_test, y_pred_svm, pos_label="spam"
)

f1_svm = f1_score(
    y_test, y_pred_svm, pos_label="spam"
)

print("\nLinear SVM Evaluation:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-Score:", f1_svm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion matrix
cm_svm = confusion_matrix(
    y_test,
    y_pred_svm,
    labels=["ham", "spam"]
)

print("\nLinear SVM Confusion Matrix:")
print(cm_svm)

# Plot confusion matrix
disp_svm = ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=["Ham", "Spam"]
)

disp_svm.plot(cmap="Blues")
plt.title("Linear SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/linear_svm_confusion_matrix.png",
    dpi=300
)

plt.show()