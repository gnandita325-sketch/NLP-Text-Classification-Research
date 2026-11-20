import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    keep_default_na=False,
    quoting=csv.QUOTE_NONE
)

print("\nfirst five records:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column information
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nNumber of Duplicate Records:")
print(df.duplicated().sum())

# Count ham and spam messages
print("\nMessage Category Counts:")
print(df["label"].value_counts())

# Calculate category percentages
print("\nMessage Category Percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Calculate message length in characters
df["character_count"] = df["message"].apply(len)

# Calculate message length in words
df["word_count"] = df["message"].apply(
    lambda message: len(message.split())
)

# Display the new columns
print("\nMessage Length Analysis:")
print(df[["label", "message", "character_count", "word_count"]].head())
# Calculate average message length for each category
print("\nAverage Message Length by Category:")

print(
    df.groupby("label")[["character_count", "word_count"]]
    .mean()
)

# Visualize average message length
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="label",
    y="character_count",
    hue="label",
    legend=False,
    errorbar=None
)

plt.title("Average Message Length: Ham vs Spam")
plt.xlabel("Message Category")
plt.ylabel("Average Number of Characters")

plt.tight_layout()
plt.savefig("results/average_message_length.png", dpi=300)
plt.show()

# Visualize average word count
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="label",
    y="word_count",
    hue="label",
    legend=False,
    errorbar=None
)

plt.title("Average Word Count: Ham vs Spam")
plt.xlabel("Message Category")
plt.ylabel("Average Number of Words")

plt.tight_layout()
plt.savefig("results/average_word_count.png", dpi=300)
plt.show()

# Visualize character count distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="character_count",
    hue="label",
    bins=40,
    kde=True,
    element="step"
)

plt.title("Message Length Distribution: Ham vs Spam")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Messages")

plt.tight_layout()
plt.savefig("results/message_length_distribution.png", dpi=300)
plt.show()

# Visualize message length using a boxplot
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="label",
    y="character_count",
    hue="label",
    legend=False
)

plt.title("Message Length Comparison: Ham vs Spam")
plt.xlabel("Message Category")
plt.ylabel("Number of Characters")

plt.tight_layout()
plt.savefig("results/message_length_boxplot.png", dpi=300)
plt.show()

# Calculate descriptive statistics for message length
print("\nCharacter Count Statistics:")
print(
    df.groupby("label")["character_count"]
    .describe()
    .to_string()
)

print("\nWord Count Statistics:")
print(
    df.groupby("label")["word_count"]
    .describe()
    .to_string()
)

# Convert messages to lowercase
df["cleaned_message"] = df["message"].apply(
    lambda message: message.lower()
)

# Display original and cleaned messages
print("\nLowercase Conversion:")
print(
    df[["message", "cleaned_message"]].head()
)

# Remove extra spaces from messages
df["cleaned_message"] = df["cleaned_message"].apply(
    lambda message: " ".join(message.split())
)

print("\nSpace Normalization:")
print(df[["message", "cleaned_message"]].head())

# Count punctuation characters in each message
import string

df["punctuation_count"] = df["cleaned_message"].apply(
    lambda message: sum(
        character in string.punctuation
        for character in message
    )
)

# Display punctuation counts
print("\nPunctuation Analysis:")
print(
    df[["cleaned_message", "punctuation_count"]].head()
)

# Compare average punctuation count by category
print("\nAverage Punctuation Count by Category:")
print(
    df.groupby("label")["punctuation_count"].mean()
)

# Check messages containing URLs
url_count = df["cleaned_message"].str.contains(
    r"http|www|\.com",
    case=False,
    regex=True
).sum()

print("\nURL Analysis:")
print("Number of messages containing URL patterns:", url_count)

# Display some messages containing URLs
print("\nSample Messages Containing URLs:")
print(
    df.loc[
        df["cleaned_message"].str.contains(
            r"http|www|\.com",
            case=False,
            regex=True
        ),
        "cleaned_message"
    ].head()
)

# Count messages containing numbers
number_count = df["cleaned_message"].str.contains(
    r"\d",
    regex=True
).sum()

print("\nNumber Analysis:")
print("Messages containing numbers:", number_count)

# Display sample messages containing numbers
print("\nSample Messages Containing Numbers:")
print(
    df.loc[
        df["cleaned_message"].str.contains(r"\d", regex=True),
        "cleaned_message"
    ].head()
)

# Check whether each message contains numbers
df["contains_number"] = df["cleaned_message"].str.contains(
    r"\d",
    regex=True
)

# Calculate percentage of messages containing numbers
print("\nNumber Usage by Message Category:")
print(
    df.groupby("label")["contains_number"].mean() * 100
)

# Load English stop words
stop_words = set(stopwords.words("english"))

# Display total number of stop words
print("\nStop Word Analysis:")
print("Total English stop words:", len(stop_words))

# Display some common stop words
print("\nSample Stop Words:")
print(list(stop_words)[:20])

# Check important negation words
print("\nNegation Words:")
for word in ["not", "no", "never"]:
    print(word, ":", word in stop_words)


from collections import Counter

# Combine all ham messages
ham_text = " ".join(
    df[df["label"] == "ham"]["cleaned_message"]
)

# Combine all spam messages
spam_text = " ".join(
    df[df["label"] == "spam"]["cleaned_message"]
)

# Count word frequencies
ham_words = Counter(ham_text.split())
spam_words = Counter(spam_text.split())

# Display most frequent words
print("\nTop 20 Ham Words:")
print(ham_words.most_common(20))

print("\nTop 20 Spam Words:")
print(spam_words.most_common(20))