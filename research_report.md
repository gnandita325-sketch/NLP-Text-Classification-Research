# Comparative Evaluation of Machine Learning Models for SMS Spam Classification Using TF-IDF

## Abstract

SMS spam detection is an important text-classification problem due to the increasing volume of unsolicited and potentially misleading messages. This study investigates the performance of three supervised machine learning algorithms: Multinomial Naive Bayes, Logistic Regression and Linear Support Vector Machine (SVM), for SMS spam classification.

The experiments use the UCI SMS Spam Collection dataset, containing 5,574 original records. After removing 403 duplicate records, 5,171 unique message-label records were retained. Text messages were represented using Term Frequency-Inverse Document Frequency (TF-IDF) features with unigrams and bigrams.

The models were evaluated using a stratified 80:20 train-test split and five-fold stratified cross-validation on the training partition. Linear SVM achieved the highest cross-validation mean accuracy of 98.69% and macro F1-score of 96.95%. On the independent test set, it achieved 98.16% accuracy and a spam F1-score of 92.24%.

Error analysis identified 18 false negatives and one false positive in the test set. These findings demonstrate the effectiveness of Linear SVM for this dataset while highlighting the importance of class-specific evaluation and further generalization testing.

## Keywords

SMS Spam Classification, Natural Language Processing, Machine Learning, TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM, Text Classification.

## 1. Introduction

Short Message Service (SMS) remains a widely used communication channel, but it is also commonly exploited for sending unsolicited promotional messages and potentially fraudulent content. Automatically distinguishing legitimate messages (ham) from spam is therefore an important binary text-classification task in Natural Language Processing (NLP).

Machine learning techniques can identify patterns in message text and use them to classify incoming messages. However, different algorithms may perform differently depending on the characteristics of the dataset and the text representation used.

In this study, the Term Frequency–Inverse Document Frequency (TF-IDF) technique is used to convert SMS messages into numerical feature vectors. Both unigrams and bigrams are considered to capture individual words as well as two-word sequences. Three machine learning algorithms—Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM)—are evaluated under consistent experimental conditions.

The study uses the UCI SMS Spam Collection dataset and compares model performance through a stratified train-test split and five-fold stratified cross-validation. In addition to standard evaluation metrics, error analysis is performed to examine incorrectly classified messages and identify potential challenges in SMS spam detection.

## 2. Research Objectives

The main objectives of this study are:

1. To explore and preprocess the UCI SMS Spam Collection dataset, including the removal of exact duplicate records.
2. To transform SMS messages into numerical representations using TF-IDF with unigram and bigram features.
3. To implement and compare Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers.
4. To evaluate model performance using accuracy, precision, recall, and F1-score.
5. To assess model consistency using five-fold stratified cross-validation on the training data.
6. To perform error analysis and examine the limitations of the best-performing model.

## 3. Methodology

### 3.1 Dataset Description

The study uses the UCI SMS Spam Collection dataset, which contains SMS messages labelled as either ham (legitimate) or spam. The original dataset contains 5,574 messages. An initial inspection identified 403 exact duplicate records. These duplicates were removed, resulting in 5,171 unique message records.

The cleaned dataset contains 4,518 ham messages and 653 spam messages. This distribution indicates class imbalance, with legitimate messages forming the majority class.

### 3.2 Data Preprocessing

The dataset was loaded and examined to identify duplicate records, missing values, and class distribution. Exact duplicate message-label pairs were removed before splitting the data.

The cleaned dataset was divided into training and testing subsets using an 80:20 stratified split, with a fixed random state of 42. Stratification preserved the approximate class proportions in both subsets.

The resulting training set contained 4,136 messages, while the test set contained 1,035 messages. The test set was kept separate from model training and cross-validation.

### 3.3 Feature Extraction

Text messages were converted into numerical feature vectors using Term Frequency–Inverse Document Frequency (TF-IDF).

The vectorizer was configured with:
- Maximum features: 5,000
- N-gram range: (1, 2), including unigrams and bigrams

Unigrams represent individual words, while bigrams represent consecutive two-word sequences. This representation allows the classifiers to use both individual terms and short word combinations when learning patterns associated with spam and legitimate messages.

### 3.4 Machine Learning Models

Three supervised machine learning algorithms were selected for comparison:

1. **Multinomial Naive Bayes:** A probabilistic classifier commonly used for text classification. It estimates class probabilities based on feature

## 4. Experimental Results

### 4.1 Held-out Test Set Performance

The three machine learning models were evaluated on the same independent test set containing 1,035 SMS messages, including 904 ham messages and 131 spam messages.

The results are summarized below:

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.81% | 100.00% | 74.81% | 85.59% |
| Logistic Regression | 96.71% | 98.99% | 74.81% | 85.22% |
| Linear SVM | **98.16%** | 99.12% | **86.26%** | **92.24%** |

Linear SVM achieved the highest accuracy, spam recall, and spam F1-score among the three models. Its spam precision was also high at 99.12%.

### 4.2 Confusion Matrix Analysis

The confusion matrices provide further insight into the classification performance.

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 904 | 0 | 33 | 98 |
| Logistic Regression | 903 | 1 | 33 | 98 |
| Linear SVM | 903 | 1 | 18 | 113 |

Linear SVM correctly identified 113 spam messages out of 131. It produced 18 false negatives and only one false positive. This indicates that it identified a larger proportion of spam messages than the other two models on the held-out test set.

### 4.3 Five-Fold Cross-Validation Results

Five-fold stratified cross-validation was performed on the training subset to examine model consistency.

| Model | Mean Accuracy | Accuracy Std. Dev. | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.74% | 0.68% | 98.20% | 87.07% | 91.60% |
| Logistic Regression | 96.47% | 0.60% | 97.95% | 86.09% | 90.84% |
| Linear SVM | **98.69%** | **0.23%** | **98.47%** | **95.56%** | **96.95%** |

Linear SVM achieved the highest mean accuracy and macro F1-score across the five folds. Its accuracy standard deviation of approximately 0.23 percentage points indicates relatively low variation across these folds.

### 4.4 Training and Inference Time

The recorded execution times were:

| Model | Training Time (seconds) | Inference Time (seconds) |
|---|---:|---:|
| Multinomial Naive Bayes | 0.01057 | 0.00057 |
| Logistic Regression | 0.05766 | 0.00053 |
| Linear SVM | 0.05169 | 0.00154 |

Inference time represents the time taken to predict all 1,035 test messages, not the time required for a single message. These are single-run measurements and should be interpreted as approximate observations rather than definitive speed comparisons.

### 4.5 Overall Result

Based on both held-out testing and cross-validation, Linear SVM demonstrated the strongest overall classification performance among the evaluated models on this dataset. It achieved 98.16% test accuracy and a spam F1-score of 92.24%, while also obtaining a mean cross-validation accuracy of 98.69% and macro F1-score of 96.95%.

These results suggest that Linear SVM is a promising choice for this particular SMS spam classification experiment. Further evaluation on additional datasets would be necessary to assess its performance on messages from other sources.

## 5. Error Analysis

Error analysis was performed on the Linear SVM model using its predictions on the independent test set of 1,035 messages.

The model produced:
- True Negatives: 903
- True Positives: 113
- False Positives: 1
- False Negatives: 18

A false positive occurs when a legitimate message is incorrectly classified as spam. A false negative occurs when a spam message is incorrectly classified as legitimate.

The model produced only one false positive, indicating that very few legitimate messages were incorrectly flagged as spam. However, 18 spam messages were classified as legitimate, showing that some spam patterns were not successfully identified.

### 5.1 Observations from Misclassified Messages

An examination of selected false-negative examples revealed several possible challenges:

1. **Unusual wording:** Some spam messages contained uncommon expressions or promotional phrases.
2. **Event-based promotions:** Messages advertising events, tickets, or special offers were sometimes difficult to distinguish from legitimate announcements.
3. **Short messages:** Very short messages may provide limited textual information for classification.
4. **Spelling variations:** Non-standard spelling and unusual text patterns may affect feature matching.
5. **Context-dependent language:** Some messages contain words that can appear in both legitimate and promotional contexts.

These observations are potential explanations based on the inspected examples. They do not establish the exact cause of each misclassification.

### 5.2 Error Analysis Summary

The results indicate that although Linear SVM performed well overall, its recall for spam messages was not perfect. Some spam messages remained difficult to identify using the selected TF-IDF representation and linear classification approach.

Future experiments could investigate additional text representations, feature engineering, alternative classification algorithms, and larger or more diverse datasets.

## 6. Discussion

The experimental results show differences in how the three selected machine learning algorithms performed on the SMS Spam Collection dataset.

Linear SVM achieved the strongest overall performance in both held-out testing and five-fold cross-validation. One possible explanation is that margin-based classification works effectively with high-dimensional, sparse TF-IDF representations. However, the experiment does not independently establish the reason for this performance difference.

Multinomial Naive Bayes achieved perfect spam precision on the test set but had lower spam recall. This means that every message it predicted as spam was spam in this test set, but it missed a larger number of actual spam messages.

Logistic Regression produced results close to Naive Bayes, but its overall performance was slightly lower in this experiment.

The cross-validation results support the held-out test findings: Linear SVM achieved the highest mean accuracy and macro F1-score, with relatively low variation across the five folds.

The error analysis also highlights an important limitation. Despite its high overall accuracy, Linear SVM missed 18 spam messages. This demonstrates why accuracy alone is insufficient for evaluating classification systems, particularly when the dataset contains imbalanced classes.

Overall, the findings indicate that Linear SVM was the strongest of the three evaluated models under the selected experimental conditions. However, the results should be interpreted within the scope of the dataset and evaluation setup used.

## 7. Limitations

Although the experiments produced encouraging results, the study has several limitations:

1. **Single dataset:** The evaluation uses only the UCI SMS Spam Collection dataset. The results may not represent messages from other platforms, regions, or time periods.
2. **Class imbalance:** The dataset contains substantially more legitimate messages than spam messages, which can influence overall evaluation results.
3. **Limited model selection:** Only three traditional machine learning algorithms were compared. Other approaches, including more advanced NLP methods, were not investigated.
4. **Single held-out split:** The final test performance is based on one stratified 80:20 split. Different data partitions may produce different results.
5. **Execution-time measurements:** Training and inference times were recorded during individual runs and may vary depending on the execution environment.
6. **Limited error investigation:** The error analysis examined selected misclassified examples but did not establish the underlying cause of every classification error.

## 8. Future Work

Several improvements can be explored in future research:

1. Evaluate the models on additional SMS and text-message datasets to examine their ability to generalize.
2. Investigate alternative text representations, including character-level features and word embeddings.
3. Explore additional machine learning and deep learning classification approaches.
4. Experiment with class-weighting and other imbalance-handling techniques to improve spam recall.
5. Conduct more detailed error analysis to understand the linguistic patterns associated with misclassified messages.
6. Perform repeated experiments with different random seeds and additional evaluation settings to obtain more robust performance estimates.

These directions could help improve classification performance and provide a broader understanding of the challenges involved in SMS spam detection.

## 9. Conclusion

This study presented a comparative evaluation of three machine learning algorithms for SMS spam classification using TF-IDF features. The UCI SMS Spam Collection dataset was examined, cleaned, and used to train and evaluate Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers.

The experimental results showed that Linear SVM achieved the best overall performance, with 98.16% accuracy and a 92.24% spam F1-score on the independent test set. It also achieved a mean five-fold cross-validation accuracy of 98.69% and a macro F1-score of 96.95%.

Error analysis revealed that the model correctly classified most messages but still missed 18 spam messages. This highlights the importance of examining class-specific metrics and misclassifications rather than relying exclusively on overall accuracy.

The findings suggest that Linear SVM is an effective approach for this particular dataset and experimental setup. Further evaluation using more diverse datasets and additional modelling techniques is needed before drawing conclusions about its broader applicability.

## 10. References

1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering*, 259–262. https://doi.org/10.1145/2034691.2034742

2. UCI Machine Learning Repository. SMS Spam Collection Dataset. University of California, Irvine. https://archive.ics.uci.edu/dataset/228/sms+spam+collection

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

4. Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. *Proceedings of the First Instructional Conference on Machine Learning*.

5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

### Software and Libraries

- Python: https://www.python.org/
- Pandas: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/
- Matplotlib: https://matplotlib.org/