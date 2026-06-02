import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# =====================================================
# Configuration
# =====================================================

# INPUT_FILE = "Eval_Results/open_ended_eval_results.csv"
INPUT_FILE = "Eval_Results/consolidated_eval_results.csv"

OUTPUT_DIR = "Eval_Results"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

LOW_THRESHOLD = 0.2

# =====================================================
# Create Output Directories
# =====================================================

os.makedirs(PLOTS_DIR, exist_ok=True)

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv(INPUT_FILE)

# Remove rows with missing correctness values
df = df.dropna(subset=["answer_correctness"])

print("=" * 60)
print(f"Total rows with valid correctness scores: {len(df)}")
print("=" * 60)

# =====================================================
# Split by Threshold
# =====================================================

low = df[df["answer_correctness"] < LOW_THRESHOLD].sort_values("answer_correctness")

high = df[df["answer_correctness"] >= LOW_THRESHOLD].sort_values(
    "answer_correctness", ascending=False
)

low.to_csv(
    f"{OUTPUT_DIR}/low_correctness_0_to_{LOW_THRESHOLD}.csv",
    index=False,
)

high.to_csv(
    f"{OUTPUT_DIR}/high_correctness_{LOW_THRESHOLD}_to_1.csv",
    index=False,
)

print(f"Low  (0.0 - {LOW_THRESHOLD}) : {len(low)} rows")

print(f"High ({LOW_THRESHOLD} - 1.0) : {len(high)} rows")

# =====================================================
# Descriptive Statistics
# =====================================================

stats = df["answer_correctness"].describe()

print("\nDescriptive Statistics:")
print(stats)

# =====================================================
# Save Evaluation Summary JSON
# =====================================================

summary_json = {
    "total_rows": int(len(df)),
    "mean": float(df["answer_correctness"].mean()),
    "median": float(df["answer_correctness"].median()),
    "std": float(df["answer_correctness"].std()),
    "min": float(df["answer_correctness"].min()),
    "max": float(df["answer_correctness"].max()),
    "q25": float(df["answer_correctness"].quantile(0.25)),
    "q75": float(df["answer_correctness"].quantile(0.75)),
    "below_0_2": int(len(df[df["answer_correctness"] < 0.2])),
    "above_0_8": int(len(df[df["answer_correctness"] >= 0.8])),
}

with open(
    f"{OUTPUT_DIR}/evaluation_summary.json",
    "w",
) as f:
    json.dump(summary_json, f, indent=4)

# =====================================================
# Bucket Distribution
# =====================================================

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

bucket_labels = [
    "0.0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
]

buckets = pd.cut(
    df["answer_correctness"],
    bins=bins,
    labels=bucket_labels,
    include_lowest=True,
)

bucket_counts = buckets.value_counts().sort_index()

distribution_json = {
    "total_rows": int(len(df)),
    "distribution": {},
}

print("\n" + "=" * 60)
print("CORRECTNESS DISTRIBUTION")
print("=" * 60)

for label, count in bucket_counts.items():

    percentage = round(
        (count / len(df)) * 100,
        2,
    )

    distribution_json["distribution"][label] = {
        "count": int(count),
        "percentage": percentage,
    }

    print(f"{label:10s} -> " f"{count:4d} rows " f"({percentage:6.2f}%)")

with open(
    f"{OUTPUT_DIR}/correctness_distribution.json",
    "w",
) as f:
    json.dump(
        distribution_json,
        f,
        indent=4,
    )

# =====================================================
# Plot 1 - Histogram
# =====================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["answer_correctness"],
    bins=20,
)

plt.title("Distribution of Answer Correctness")
plt.xlabel("Correctness Score")
plt.ylabel("Number of Questions")

plt.grid(True)

plt.savefig(
    f"{PLOTS_DIR}/1_histogram.png",
    bbox_inches="tight",
)

plt.show()

# =====================================================
# Plot 2 - Box Plot
# =====================================================

plt.figure(figsize=(10, 3))

plt.boxplot(
    df["answer_correctness"],
    vert=False,
)

plt.title("Answer Correctness Box Plot")
plt.xlabel("Correctness Score")

plt.grid(True)

plt.savefig(
    f"{PLOTS_DIR}/2_boxplot.png",
    bbox_inches="tight",
)

plt.show()

# =====================================================
# Plot 3 - Cumulative Distribution
# =====================================================

scores = np.sort(df["answer_correctness"])

cdf = np.arange(
    1,
    len(scores) + 1,
) / len(scores)

plt.figure(figsize=(10, 6))

plt.plot(
    scores,
    cdf,
    linewidth=2,
)

plt.title("Cumulative Distribution of Correctness")

plt.xlabel("Correctness Score")
plt.ylabel("Fraction of Questions")

plt.grid(True)

plt.savefig(
    f"{PLOTS_DIR}/3_cdf.png",
    bbox_inches="tight",
)

plt.show()

# =====================================================
# Plot 4 - Quality Buckets
# =====================================================

quality_labels = [
    "<0.2",
    "0.2-0.4",
    "0.4-0.6",
    "0.6-0.8",
    ">0.8",
]

quality_counts = [
    len(df[df["answer_correctness"] < 0.2]),
    len(df[(df["answer_correctness"] >= 0.2) & (df["answer_correctness"] < 0.4)]),
    len(df[(df["answer_correctness"] >= 0.4) & (df["answer_correctness"] < 0.6)]),
    len(df[(df["answer_correctness"] >= 0.6) & (df["answer_correctness"] < 0.8)]),
    len(df[df["answer_correctness"] >= 0.8]),
]

plt.figure(figsize=(10, 6))

plt.bar(
    quality_labels,
    quality_counts,
)

plt.title("RAG Answer Quality Buckets")
plt.xlabel("Correctness Range")
plt.ylabel("Number of Questions")

for i, count in enumerate(quality_counts):
    plt.text(
        i,
        count + 5,
        str(count),
        ha="center",
    )

plt.savefig(
    f"{PLOTS_DIR}/4_quality_buckets.png",
    bbox_inches="tight",
)

plt.show()

# =====================================================
# Plot 5 - Bucket Distribution
# =====================================================

plt.figure(figsize=(12, 6))

bucket_counts.plot(kind="bar")

plt.title("Correctness Distribution (0.1 Buckets)")

plt.xlabel("Correctness Range")
plt.ylabel("Number of Questions")

for i, count in enumerate(bucket_counts):
    plt.text(
        i,
        count + 5,
        str(count),
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    f"{PLOTS_DIR}/5_bucket_distribution.png",
    bbox_inches="tight",
)

plt.show()

# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)

print(f"{OUTPUT_DIR}/low_correctness_0_to_{LOW_THRESHOLD}.csv")
print(f"{OUTPUT_DIR}/high_correctness_{LOW_THRESHOLD}_to_1.csv")

print(f"{OUTPUT_DIR}/evaluation_summary.json")

print(f"{OUTPUT_DIR}/correctness_distribution.json")

print(f"{PLOTS_DIR}/1_histogram.png")

print(f"{PLOTS_DIR}/2_boxplot.png")

print(f"{PLOTS_DIR}/3_cdf.png")

print(f"{PLOTS_DIR}/4_quality_buckets.png")

print(f"{PLOTS_DIR}/5_bucket_distribution.png")

print("=" * 60)
