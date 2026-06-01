import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv("Eval_Results/open_ended_eval_results.csv")

print("=" * 60)
print(f"Total rows: {len(df)}")
print("=" * 60)

# =====================================================
# Split by Threshold
# =====================================================

MIN = 0.2

low = df[df["answer_correctness"] < MIN].sort_values("answer_correctness")

high = df[df["answer_correctness"] >= MIN].sort_values(
    "answer_correctness", ascending=False
)

low.to_csv(f"Eval_Results/low_correctness_0_to_{MIN}.csv", index=False)

high.to_csv(f"Eval_Results/high_correctness_{MIN}_to_1.csv", index=False)

print(f"Low  (0.0 - {MIN}) : {len(low)} rows")

print(f"High ({MIN} - 1.0) : {len(high)} rows")

print("\nDescriptive Statistics:")
print(df["answer_correctness"].describe())

print("\nHistogram Buckets:")
print(df["answer_correctness"].value_counts(bins=10))

# =====================================================
# Create output folder
# =====================================================

os.makedirs("Eval_Results/plots", exist_ok=True)

# =====================================================
# Plot 1 - Histogram
# =====================================================

plt.figure(figsize=(10, 6))

plt.hist(df["answer_correctness"], bins=20)

plt.title("Distribution of Answer Correctness")
plt.xlabel("Correctness Score")
plt.ylabel("Number of Questions")

plt.grid(True)

plt.savefig("Eval_Results/plots/1_histogram.png", bbox_inches="tight")

plt.show()

# =====================================================
# Plot 2 - Box Plot
# =====================================================

plt.figure(figsize=(10, 3))

plt.boxplot(df["answer_correctness"], vert=False)

plt.title("Answer Correctness Box Plot")
plt.xlabel("Correctness Score")

plt.grid(True)

plt.savefig("Eval_Results/plots/2_boxplot.png", bbox_inches="tight")

plt.show()

# =====================================================
# Plot 3 - Bucket Bar Chart
# =====================================================

bucket_counts = df["answer_correctness"].value_counts(bins=10).sort_index()

plt.figure(figsize=(12, 6))

bucket_counts.plot(kind="bar")

plt.title("Correctness Score Buckets")
plt.xlabel("Score Range")
plt.ylabel("Number of Questions")

plt.tight_layout()

plt.savefig("Eval_Results/plots/3_bucket_bar_chart.png", bbox_inches="tight")

plt.show()

# =====================================================
# Plot 4 - Cumulative Distribution
# =====================================================

scores = np.sort(df["answer_correctness"])

cdf = np.arange(1, len(scores) + 1) / len(scores)

plt.figure(figsize=(10, 6))

plt.plot(scores, cdf, linewidth=2)

plt.title("Cumulative Distribution of Correctness")
plt.xlabel("Correctness Score")
plt.ylabel("Fraction of Questions")

plt.grid(True)

plt.savefig("Eval_Results/plots/4_cdf.png", bbox_inches="tight")

plt.show()

# =====================================================
# Plot 5 - Quality Buckets
# =====================================================

labels = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"]

counts = [
    len(df[df["answer_correctness"] < 0.2]),
    len(df[(df["answer_correctness"] >= 0.2) & (df["answer_correctness"] < 0.4)]),
    len(df[(df["answer_correctness"] >= 0.4) & (df["answer_correctness"] < 0.6)]),
    len(df[(df["answer_correctness"] >= 0.6) & (df["answer_correctness"] < 0.8)]),
    len(df[df["answer_correctness"] >= 0.8]),
]

plt.figure(figsize=(10, 6))

plt.bar(labels, counts)

plt.title("RAG Answer Quality Buckets")
plt.xlabel("Correctness Range")
plt.ylabel("Number of Questions")

for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha="center")

plt.savefig("Eval_Results/plots/5_quality_buckets.png", bbox_inches="tight")

plt.show()

# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)
print("Plots Saved To:")
print("Eval_Results/plots/")
print("=" * 60)

print("1_histogram.png")
print("2_boxplot.png")
print("3_bucket_bar_chart.png")
print("4_cdf.png")
print("5_quality_buckets.png")
print("=" * 60)
