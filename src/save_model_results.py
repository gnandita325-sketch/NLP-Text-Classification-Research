
import pandas as pd

# Consolidated model evaluation results
results = {
    "Model": [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Linear SVM"
    ],

    "Accuracy": [
        0.9681159420,
        0.9671497585,
        0.9816425121
    ],

    "Precision": [
        1.0000000000,
        0.9898989899,
        0.9912280702
    ],

    "Recall": [
        0.7480916031,
        0.7480916031,
        0.8625954198
    ],

    "F1_Score": [
        0.8558951965,
        0.8521739130,
        0.9224489796
    ],

    "Training_Time": [
        0.0105739001,
        0.0576607999,
        0.0516932001
    ],

    "Inference_Time": [
        0.0005743999,
        0.0005266999,
        0.0015388001
    ]
}

# Convert dictionary into DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("Model comparison results saved successfully!")

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))
