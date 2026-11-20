import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results/model_comparison.csv")

print("Model comparison data loaded successfully!")
print(df)
plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["Accuracy"] * 100)

plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.ylim(90, 100)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("results/model_accuracy_comparison.png", dpi=300)

plt.show()

plt.figure(figsize=(9, 5))

x = range(len(df["Model"]))
width = 0.25

plt.bar(
    [i - width for i in x],
    df["Precision"] * 100,
    width,
    label="Precision"
)

plt.bar(
    x,
    df["Recall"] * 100,
    width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    df["F1_Score"] * 100,
    width,
    label="F1-Score"
)

plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score (%)")

plt.xticks(list(x), df["Model"], rotation=15)
plt.ylim(0, 110)

plt.legend()
plt.tight_layout()

plt.savefig("results/model_metrics_comparison.png", dpi=300)

plt.show()

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Training_Time"] * 1000
)

plt.title("Training Time Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Training Time (milliseconds)")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("results/model_training_time_comparison.png", dpi=300)

plt.show()