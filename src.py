# %%
# EDA for Fake/True News Dataset


import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Plot style
sns.set(context="notebook", style="whitegrid")

# Paths
PROJECT_ROOT = Path("/cse250")
DATA_DIR = PROJECT_ROOT / "archive"
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

print(f"Fake path exists: {FAKE_PATH.exists()} | True path exists: {TRUE_PATH.exists()}")


# %%
# Load data
fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print("Shapes:")
print({"fake": fake_df.shape, "true": true_df.shape})

display(fake_df.head(3))
display(true_df.head(3))


# %%
# Normalize column names (lowercase) for safety and add labels
fake_df.columns = [c.lower() for c in fake_df.columns]
true_df.columns = [c.lower() for c in true_df.columns]

fake_df["label"] = "fake"
true_df["label"] = "true"

# Combine
combined_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
print("Combined shape:", combined_df.shape)


display(combined_df.sample(min(5, len(combined_df)), random_state=42))
print("Columns:", list(combined_df.columns))


# %%
# Basic info, types, and memory usage
import io
buffer = io.StringIO()
combined_df.info(buf=buffer)
print(buffer.getvalue())

# Summary statistics for numeric columns
num_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    display(combined_df[num_cols].describe().T)
else:
    print("No numeric columns found; skipping numeric describe.")

# Distinct counts for object columns
obj_cols = combined_df.select_dtypes(include=["object"]).columns
if len(obj_cols) > 0:
    display(combined_df[obj_cols].describe().T)
else:
    print("No object (string) columns found")


# %%
# Missing values and duplicates
missing_counts = combined_df.isna().sum().sort_values(ascending=False)
missing_pct = (missing_counts / len(combined_df) * 100).round(2)
missing_df = pd.DataFrame({"missing": missing_counts, "missing_%": missing_pct})
display(missing_df[missing_df["missing"] > 0])

# Duplicates across all columns
dup_rows = combined_df.duplicated().sum()
print(f"Duplicate rows (all columns): {dup_rows}")

combined_nodup_df = combined_df.drop_duplicates().reset_index(drop=True)
print("Shape after dropping duplicates:", combined_nodup_df.shape)


# %%
# Label distribution
plt.figure(figsize=(5,4))
ax = sns.countplot(data=combined_df, x="label", order=combined_df["label"].value_counts().index)
plt.title("Label Distribution")
plt.xlabel("")
plt.ylabel("Count")
for c in ax.containers:
    ax.bar_label(c)
plt.show()


# %%
# Text-specific EDA 

def first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

text_col = first_present(combined_df, ["text", "content", "article", "body"])
title_col = first_present(combined_df, ["title", "headline"])
subject_col = first_present(combined_df, ["subject", "category", "topic"])
date_col = first_present(combined_df, ["date", "published", "publish_date"]) 

print({"text_col": text_col, "title_col": title_col, "subject_col": subject_col, "date_col": date_col})

# Create simple text length features if text exists
if text_col is not None:
    work_df = combined_nodup_df.copy()
    work_df[text_col] = work_df[text_col].astype(str)
    work_df["char_count"] = work_df[text_col].str.len()
    work_df["word_count"] = work_df[text_col].str.split().apply(len)
else:
    work_df = combined_nodup_df.copy()

work_df.head(3)


# %%
# Distribution of text lengths by label
if text_col is not None:

    len_df = work_df[["char_count", "word_count", "label"]].copy()
    len_df["char_count"] = pd.to_numeric(len_df["char_count"], errors="coerce")
    len_df["word_count"] = pd.to_numeric(len_df["word_count"], errors="coerce")
    len_df = len_df.replace([np.inf, -np.inf], np.nan).dropna()
    len_df = len_df[(len_df["char_count"] >= 0) & (len_df["word_count"] >= 0)]

    if not len_df.empty:
        for col in ["char_count", "word_count"]:
            hi = len_df[col].quantile(0.995)
            len_df[col] = np.clip(len_df[col], 0, hi)

    max_rows = 200000
    if len_df.shape[0] > max_rows:
        len_df = len_df.sample(max_rows, random_state=42)

    if len_df.empty:
        print("Length features are empty after cleaning; skipping plots.")
    else:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(data=len_df, x="char_count", hue="label", bins=50, element="step", stat="density", common_norm=False, ax=axes[0])
            axes[0].set_title("Character Count by Label")
            axes[0].set_xlabel("Characters per article")
            axes[0].set_ylabel("Density")
            axes[0].set_xlim(left=0)

            sns.histplot(data=len_df, x="word_count", hue="label", bins=50, element="step", stat="density", common_norm=False, ax=axes[1])
            axes[1].set_title("Word Count by Label")
            axes[1].set_xlabel("Words per article")
            axes[1].set_ylabel("Density")
            axes[1].set_xlim(left=0)
            plt.tight_layout(rect=(0, 0.05, 1, 1))
            plt.figtext(0.5, 0.01, "Histograms show density; x-axes clipped at 99.5th percentile", ha="center", fontsize=9)
            plt.show()
        except Exception as e:
            # Fallback to matplotlib-only plotting per label
            labels = sorted(len_df["label"].unique())
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = plt.cm.Set2.colors
            for i, lbl in enumerate(labels):
                sub = len_df[len_df["label"] == lbl]
                axes[0].hist(sub["char_count"], bins=50, alpha=0.5, density=True, label=str(lbl), color=colors[i % len(colors)])
                axes[1].hist(sub["word_count"], bins=50, alpha=0.5, density=True, label=str(lbl), color=colors[i % len(colors)])
            axes[0].set_title("Character Count by Label")
            axes[1].set_title("Word Count by Label")
            axes[0].set_xlim(left=0)
            axes[1].set_xlim(left=0)
            axes[0].legend(title="label")
            axes[1].legend(title="label")
            plt.tight_layout(rect=(0, 0.05, 1, 1))
            plt.figtext(0.5, 0.01, "Histograms show density; x-axes clipped at 99.5th percentile", ha="center", fontsize=9)
            plt.show()
else:
    print("No text column found; skipping text length distributions.")


# %%
# Top subjects/categories 
if subject_col is not None:
    plt.figure(figsize=(8, 5))
    top_subjects = (work_df[subject_col]
                    .fillna("<NA>")
                    .value_counts()
                    .head(20)
                   )
    sns.barplot(x=top_subjects.values, y=top_subjects.index, orient="h")
    plt.title("Top Subjects / Categories")
    plt.xlabel("Count")
    plt.ylabel(subject_col)
    plt.tight_layout()
    plt.show()
else:
    print("No subject/category column found; skipping subject distribution.")


# %%
# Publishing timeline by label (if date exists)
if date_col is not None:
    timeline_df = work_df.copy()
    # Robust date parsing with multiple attempts
    parsed = pd.to_datetime(timeline_df[date_col], errors="coerce")
    remaining = timeline_df[parsed.isna()][date_col].astype(str)

    def try_parse(series, fmt=None, dayfirst=False):
        try:
            return pd.to_datetime(series, errors="coerce", format=fmt, dayfirst=dayfirst)
        except Exception:
            return pd.Series([pd.NaT] * len(series), index=series.index)

    if remaining.shape[0] > 0:
        # Common long/short month formats
        alt1 = try_parse(remaining, fmt="%B %d, %Y")   # e.g., December 31, 2017
        alt2 = try_parse(remaining, fmt="%b %d, %Y")    # e.g., Dec 31, 2017
        alt3 = try_parse(remaining, fmt="%Y-%m-%d")     # e.g., 2017-12-31
        alt4 = try_parse(remaining, fmt="%m/%d/%Y")     # e.g., 12/31/2017
        alt5 = try_parse(remaining, dayfirst=True)       # try day-first
        parsed.loc[remaining.index] = alt1.fillna(alt2).fillna(alt3).fillna(alt4).fillna(alt5)

    timeline_df["_parsed_date"] = parsed
    valid = timeline_df.dropna(subset=["_parsed_date"]).copy()

    if len(valid) > 0:
        # Normalize to midnight and ensure datetime64[ns]
        valid["_date"] = pd.to_datetime(valid["_parsed_date"]).dt.normalize()

        # Aggregate to weekly counts for readability and to align missing days
        weekly = (valid
                  .groupby([pd.Grouper(key="_date", freq="W"), "label"]) 
                  .size()
                  .reset_index(name="count")
                  .sort_values("_date"))

        # Report any labels with zero parsed dates
        present_labels = set(weekly["label"].unique())
        expected_labels = set(work_df["label"].unique())
        missing = expected_labels - present_labels
        if missing:
            print(f"Warning: no parseable dates for labels: {sorted(missing)}")

        try:
            plt.figure(figsize=(12, 4))
            sns.lineplot(data=weekly, x="_date", y="count", hue="label", marker="o")
            plt.title("Weekly Article Counts by Label")
            plt.xlabel("Week")
            plt.ylabel("Articles per week")
            plt.tight_layout()
            plt.show()
        except Exception:
            # Fallback: pivot and plot with matplotlib
            pivot = weekly.pivot(index="_date", columns="label", values="count").fillna(0)
            ax = pivot.plot(figsize=(12, 4), marker="o")
            ax.set_title("Weekly Article Counts by Label")
            ax.set_xlabel("Week")
            ax.set_ylabel("Articles per week")
            plt.tight_layout()
            plt.show()
    else:
        print("Date column present but not parseable; skipping timeline.")
else:
    print("No date column found; skipping timeline plot.")


# %%
# Most frequent words by label (quick + simple, no external stopwords)
# This is a lightweight approximation
from collections import Counter

BASIC_STOPWORDS = {
    "the","a","an","and","or","but","if","in","on","at","to","for","of","by","with",
    "is","are","was","were","be","been","being","as","that","this","it","its","from","not",
    "you","your","yours","we","our","ours","they","their","theirs","he","she","his","her","them",
}

if text_col is not None:
    def tokenize(text: str):
        return [t.strip(".,!?;:\"'()[]{}<>/\\|`~@#$%^&*-_=+").lower() for t in text.split()]

    counters = {}
    for label, grp in work_df.groupby("label"):
        words = []
        for t in grp[text_col].astype(str).tolist():
            words.extend([w for w in tokenize(t) if w and w not in BASIC_STOPWORDS and len(w) > 2 and not w.isdigit()])
        counters[label] = Counter(words)

    n = 20
    rows = []
    for label, ctr in counters.items():
        for word, cnt in ctr.most_common(n):
            rows.append({"label": label, "word": word, "count": cnt})
    top_words_df = pd.DataFrame(rows)

    if len(top_words_df) > 0:
        g = sns.catplot(data=top_words_df, x="count", y="word", col="label", kind="bar", col_wrap=2, sharex=False, height=5)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Top Words by Label")
        plt.show()
    else:
        print("No words extracted; possibly empty text column.")
else:
    print("No text column found; skipping top words.")


# %%
# 1. Subject/Category Distribution by Label
# This shows which subjects are more common in fake vs true news

if subject_col is not None:
    print("=== Subject/Category Distribution by Label ===\n")
    
    # Cross-tabulation: subject vs label
    crosstab = pd.crosstab(work_df[subject_col].fillna("Unknown"), work_df["label"], margins=True)
    print("Cross-tabulation (Subject × Label):")
    display(crosstab)
    
    # Percentage distribution
    crosstab_pct = pd.crosstab(work_df[subject_col].fillna("Unknown"), work_df["label"], normalize="index") * 100
    crosstab_pct = crosstab_pct.round(1)
    print("\nPercentage distribution (each row sums to 100%):")
    display(crosstab_pct)
    
    # Visualize: stacked bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot by subject and label
    top_subjects = work_df[subject_col].fillna("Unknown").value_counts().head(10).index
    subset = work_df[work_df[subject_col].fillna("Unknown").isin(top_subjects)]
    
    crosstab_viz = pd.crosstab(subset[subject_col].fillna("Unknown"), subset["label"])
    crosstab_viz.plot(kind="bar", stacked=True, ax=axes[0], color=["#ff6b6b", "#4ecdc4"])
    axes[0].set_title("Article Counts by Subject and Label (Top 10)")
    axes[0].set_xlabel(subject_col)
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Label")
    axes[0].tick_params(axis="x", rotation=45)
    
    # Percentage plot
    crosstab_viz_pct = crosstab_viz.div(crosstab_viz.sum(axis=1), axis=0) * 100
    crosstab_viz_pct.plot(kind="bar", stacked=True, ax=axes[1], color=["#ff6b6b", "#4ecdc4"])
    axes[1].set_title("Percentage Distribution by Subject (Top 10)")
    axes[1].set_xlabel(subject_col)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Label")
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.show()
else:
    print("No subject/category column found; skipping subject distribution analysis.")


# %%
# Save combined and deduplicated dataset for downstream tasks
OUTPUT_CSV = PROJECT_ROOT / "combined_news.csv"
work_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved combined dataset to: {OUTPUT_CSV}")


# %% [markdown]
# # Naïve Bayes Classifier Implementation
# 
# This section implements a Naïve Bayes classifier for fake news detection 
# 

# %%
# Import necessary libraries for Naïve Bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# %%
# Prepare the dataset for classification

# Combine title and text for better text features
if 'title' in work_df.columns and 'text' in work_df.columns:
    work_df['combined_text'] = work_df['title'].astype(str) + ' ' + work_df['text'].astype(str)
else:
    work_df['combined_text'] = work_df[text_col].astype(str)

# Remove rows with empty text
work_df = work_df[work_df['combined_text'].str.strip().str.len() > 0].copy()

# Encode labels: fake=0, true=1
le = LabelEncoder()
work_df['label_encoded'] = le.fit_transform(work_df['label'])

print(f"Dataset shape: {work_df.shape}")
print(f"Label distribution:\n{work_df['label'].value_counts()}")
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# %%
# Split data into train and test sets (80/20 split)
X_text = work_df['combined_text'].values
y = work_df['label_encoded'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train_text)}")
print(f"Test set size: {len(X_test_text)}")
print(f"\nTraining label distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Test label distribution:\n{pd.Series(y_test).value_counts().sort_index()}")


# %%
# Text Vectorization using TF-IDF

print("Vectorizing text with TF-IDF...")


vectorizer = TfidfVectorizer(
    max_features=10000,  # Top 10,000 features
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,  # Word must appear in at least 2 documents
    max_df=0.95,  # Ignore words that appear in >95% of documents
    stop_words='english',  # Remove English stopwords
    lowercase=True,
    strip_accents='unicode'
)

# Fit on training data and transform both train and test
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"Training TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF matrix shape: {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# %%
# Train Multinomial Naïve Bayes classifier

print("Training Multinomial Naïve Bayes classifier...")

nb_classifier = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
nb_classifier.fit(X_train_tfidf, y_train)

print("Training completed!")


# %%
# Make predictions
y_train_pred = nb_classifier.predict(X_train_tfidf)
y_test_pred = nb_classifier.predict(X_test_tfidf)
y_test_proba = nb_classifier.predict_proba(X_test_tfidf)[:, 1]  # Probability of class 1 (true)

print("Predictions generated!")


# %%
# Comprehensive Evaluation Metrics

print("=" * 60)
print("NAÏVE BAYES CLASSIFIER - EVALUATION RESULTS")
print("=" * 60)

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Per-class metrics
test_precision_per_class = precision_score(y_test, y_test_pred, average=None)
test_recall_per_class = recall_score(y_test, y_test_pred, average=None)
test_f1_per_class = f1_score(y_test, y_test_pred, average=None)

# ROC AUC
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n📊 TRAINING SET METRICS:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print("\n📊 TEST SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC AUC:   {test_auc:.4f}")

print("\n📊 PER-CLASS METRICS (Test Set):")
class_names = le.classes_
for i, class_name in enumerate(class_names):
    print(f"\n  {class_name.upper()} (Class {i}):")
    print(f"    Precision: {test_precision_per_class[i]:.4f}")
    print(f"    Recall:    {test_recall_per_class[i]:.4f}")
    print(f"    F1-Score:  {test_f1_per_class[i]:.4f}")

print("\n" + "=" * 60)


# %%
# Detailed Classification Report
print("\n📋 DETAILED CLASSIFICATION REPORT (Test Set):\n")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))


# %%
# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training confusion matrix
cm_train = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_title('Confusion Matrix - Training Set')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Test confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title('Confusion Matrix - Test Set')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Print confusion matrix values
print("Training Set Confusion Matrix:")
print(cm_train)
print("\nTest Set Confusion Matrix:")
print(cm_test)


# %%
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Feature Importance Analysis: Top words for each class

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature names (words/ngrams)
feature_names = vectorizer.get_feature_names_out()


log_probs_fake = nb_classifier.feature_log_prob_[0]  # Class 0 (fake)
log_probs_true = nb_classifier.feature_log_prob_[1]  # Class 1 (true)


log_odds = log_probs_true - log_probs_fake

# Get top words for each class
n_top = 30

# Top words for "fake" class (most negative log odds)
top_fake_indices = np.argsort(log_odds)[:n_top]
top_fake_words = [feature_names[i] for i in top_fake_indices]
top_fake_scores = log_odds[top_fake_indices]

# Top words for "true" class (most positive log odds)
top_true_indices = np.argsort(log_odds)[-n_top:][::-1]
top_true_words = [feature_names[i] for i in top_true_indices]
top_true_scores = log_odds[top_true_indices]

print(f"\nTOP {n_top} WORDS INDICATIVE OF FAKE NEWS:")
print("-" * 60)
for word, score in zip(top_fake_words, top_fake_scores):
    print(f"  {word:30s} (log odds: {score:8.4f})")

print(f"\nTOP {n_top} WORDS INDICATIVE OF TRUE NEWS:")
print("-" * 60)
for word, score in zip(top_true_words, top_true_scores):
    print(f"  {word:30s} (log odds: {score:8.4f})")


# %%
# Visualize top words for each class
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Top words for fake news
fake_df_viz = pd.DataFrame({
    'word': top_fake_words[:20],
    'log_odds': top_fake_scores[:20]
})
sns.barplot(data=fake_df_viz, y='word', x='log_odds', ax=axes[0], palette='Reds_r')
axes[0].set_title(f'Top 20 Words Indicative of FAKE News\n(Negative log odds)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Log Odds Ratio (lower = more indicative of fake)')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)

# Top words for true news
true_df_viz = pd.DataFrame({
    'word': top_true_words[:20],
    'log_odds': top_true_scores[:20]
})
sns.barplot(data=true_df_viz, y='word', x='log_odds', ax=axes[1], palette='Greens_r')
axes[1].set_title(f'Top 20 Words Indicative of TRUE News\n(Positive log odds)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log Odds Ratio (higher = more indicative of true)')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()


# %%
# Model Comparison: Try different Naïve Bayes variants
print("=" * 60)
print("COMPARING DIFFERENT NAÏVE BAYES VARIANTS")
print("=" * 60)


results = {}

# MultinomialNB with TF-IDF (current)
results['MultinomialNB (TF-IDF)'] = {
    'accuracy': test_accuracy,
    'f1': test_f1,
    'auc': test_auc
}

# ComplementNB with TF-IDF
print("\nTraining ComplementNB with TF-IDF...")
complement_nb = ComplementNB(alpha=1.0)
complement_nb.fit(X_train_tfidf, y_train)
y_pred_complement = complement_nb.predict(X_test_tfidf)
y_proba_complement = complement_nb.predict_proba(X_test_tfidf)[:, 1]

results['ComplementNB (TF-IDF)'] = {
    'accuracy': accuracy_score(y_test, y_pred_complement),
    'f1': f1_score(y_test, y_pred_complement, average='weighted'),
    'auc': roc_auc_score(y_test, y_proba_complement)
}

# MultinomialNB with CountVectorizer
print("Training MultinomialNB with CountVectorizer...")
count_vectorizer = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english',
    lowercase=True,
    strip_accents='unicode'
)
X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

nb_count = MultinomialNB(alpha=1.0)
nb_count.fit(X_train_count, y_train)
y_pred_count = nb_count.predict(X_test_count)
y_proba_count = nb_count.predict_proba(X_test_count)[:, 1]

results['MultinomialNB (Count)'] = {
    'accuracy': accuracy_score(y_test, y_pred_count),
    'f1': f1_score(y_test, y_pred_count, average='weighted'),
    'auc': roc_auc_score(y_test, y_proba_count)
}

# Display comparison
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print("\n📊 MODEL COMPARISON:")
print(comparison_df.to_string())


# %%
# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = ['accuracy', 'f1', 'auc']
titles = ['Accuracy', 'F1-Score', 'ROC AUC']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    comparison_df[metric].plot(kind='bar', ax=axes[i], color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[i].set_title(f'{title} Comparison', fontweight='bold')
    axes[i].set_ylabel(title)
    axes[i].set_xlabel('Model')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3, axis='y')
    axes[i].set_ylim([0.8, 1.0]) 
    
    # Add value labels on bars
    for j, v in enumerate(comparison_df[metric]):
        axes[i].text(j, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# %%
# Error Analysis: Examine misclassified examples
print("=" * 60)
print("ERROR ANALYSIS: MISCLASSIFIED EXAMPLES")
print("=" * 60)

# Get misclassified examples
misclassified_mask = y_test != y_test_pred
misclassified_indices = np.where(misclassified_mask)[0]

print(f"\nTotal misclassified examples: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")

# Analyze false positives (predicted true but actually fake)
false_positives = misclassified_indices[(y_test[misclassified_indices] == 0) & (y_test_pred[misclassified_indices] == 1)]
print(f"\nFalse Positives (Predicted TRUE, Actually FAKE): {len(false_positives)}")

# Analyze false negatives (predicted fake but actually true)
false_negatives = misclassified_indices[(y_test[misclassified_indices] == 1) & (y_test_pred[misclassified_indices] == 0)]
print(f"False Negatives (Predicted FAKE, Actually TRUE): {len(false_negatives)}")

print("\n" + "-" * 60)
print("SAMPLE FALSE POSITIVES (Predicted TRUE, Actually FAKE):")
print("-" * 60)
for i, idx in enumerate(false_positives[:5]):
    text_sample = X_test_text[idx][:200] + "..." if len(X_test_text[idx]) > 200 else X_test_text[idx]
    print(f"\nExample {i+1}:")
    print(f"  Text preview: {text_sample}")
    print(f"  Predicted probability (TRUE): {y_test_proba[idx]:.4f}")

print("\n" + "-" * 60)
print("SAMPLE FALSE NEGATIVES (Predicted FAKE, Actually TRUE):")
print("-" * 60)
for i, idx in enumerate(false_negatives[:5]):
    text_sample = X_test_text[idx][:200] + "..." if len(X_test_text[idx]) > 200 else X_test_text[idx]
    print(f"\nExample {i+1}:")
    print(f"  Text preview: {text_sample}")
    print(f"  Predicted probability (TRUE): {y_test_proba[idx]:.4f}")



