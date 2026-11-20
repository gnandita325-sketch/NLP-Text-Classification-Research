# SMS Spam Classification Using Machine Learning

## Overview

This project presents a comparative study of three machine learning algorithms for SMS spam classification using Natural Language Processing (NLP) and TF-IDF feature extraction.

The study evaluates Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM) under consistent experimental conditions using the UCI SMS Spam Collection dataset.

It also includes exploratory data analysis, five-fold stratified cross-validation, model evaluation, and error analysis.

## Research Objectives

- Explore and preprocess the SMS Spam Collection dataset.
- Extract textual features using TF-IDF with unigrams and bigrams.
- Compare three traditional machine learning classifiers.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Examine model consistency through cross-validation.
- Investigate classification errors and model limitations.

## Dataset

**Source:** UCI SMS Spam Collection

The original dataset contains 5,574 messages. After removing 403 exact duplicate records, 5,171 unique messages were retained.

| Class | Number of Messages |
|---|---:|
| Ham | 4,518 |
| Spam | 653 |
| Total | 5,171 |

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Methodology

1. Dataset inspection and exploratory data analysis.
2. Removal of exact duplicate records.
3. Stratified 80:20 train-test split.
4. TF-IDF feature extraction with a maximum of 5,000 features and unigram-bigram combinations.
5. Training and comparison of three classifiers.
6. Five-fold stratified cross-validation on the training partition.
7. Held-out test evaluation and error analysis.

## Models Evaluated

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

## Experimental Results

### Held-out Test Performance

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.81% | 100.00% | 74.81% | 85.59% |
| Logistic Regression | 96.71% | 98.99% | 74.81% | 85.22% |
| Linear SVM | **98.16%** | 99.12% | **86.26%** | **92.24%** |

### Five-Fold Cross-Validation

| Model | Mean Accuracy | Macro F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 96.74% | 91.60% |
| Logistic Regression | 96.47% | 90.84% |
| Linear SVM | **98.69%** | **96.95%** |

Linear SVM achieved the strongest overall results in this experiment.

## Error Analysis

The Linear SVM model produced the following results on the held-out test set:

- True Positives: 113
- True Negatives: 903
- False Positives: 1
- False Negatives: 18

The analysis identified possible challenges involving promotional language, unusual wording, short messages, and spelling variations.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Natural Language Processing
- TF-IDF
- Machine Learning

## Project Structure

## Project Structure

```text
NLP-Text-Classification-Research/
│
├── data/
│   └── SMSSpamCollection
│
├── notebooks/
│   └── 01_data_exploration.py
│
├── results/
│   ├── eda_summary.txt
│   ├── error_analysis_summary.txt
│   ├── research_discussion.txt
│   ├── cross_validation_results.csv
│   ├── model_comparison.csv
│   ├── average_message_length.png
│   ├── average_word_count.png
│   ├── message_length_boxplot.png
│   ├── message_length_distribution.png
│   ├── naive_bayes_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── linear_svm_confusion_matrix.png
│   ├── model_accuracy_comparison.png
│   ├── model_metrics_comparison.png
│   └── model_training_time_comparison.png
│
├── src/
│   ├── cross_validation.py
│   ├── error_analysis.py
│   ├── plot_model_comparison.py
│   ├── save_model_results.py
│   └── text_preprocessing.py
│
├── research_report.md
├── requirements.txt
├── README.md
└── .gitignore
```

The virtual environment (`venv/`) is excluded from version control.

## Research Report

The detailed research report is available in `research_report.md`. It covers the methodology, experimental results, discussion, limitations, and future research directions.

## Limitations and Future Work

The study uses one dataset and a single held-out split. Future work could explore additional datasets, alternative text representations, class-imbalance techniques, and more advanced NLP models.

## Author

**G Nandita**

B.Tech Computer Science and Engineering (AI/ML)

NIST University