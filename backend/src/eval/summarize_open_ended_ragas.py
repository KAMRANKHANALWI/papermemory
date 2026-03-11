import json
import pandas as pd
from pathlib import Path

# ---------------------------------------
# Paths
# ---------------------------------------
RESULTS_DIR = Path("results")
SUMMARY_DIR = Path("results")

CSV_PATTERN = "open_ended_eval_results_*.csv"

csv_files = list(RESULTS_DIR.glob(CSV_PATTERN))

if not csv_files:
    raise FileNotFoundError(
        f"No open-ended result files found matching pattern: {CSV_PATTERN}"
    )

print(f"Found {len(csv_files)} open-ended result file(s):")
for f in csv_files:
    print(" -", f.name)

# ---------------------------------------
# Metric columns (ONLY what we trust)
# ---------------------------------------
METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


# ---------------------------------------
# Helper: summarize one CSV
# ---------------------------------------
def summarize_open_ended(csv_path: Path):
    df = pd.read_csv(csv_path)

    print(f"\nProcessing: {csv_path.name}")
    print("Detected columns:", df.columns.tolist())

    total = len(df)

    # ---------------------------------------
    # Drop error column if present
    # ---------------------------------------
    if "error" in df.columns:
        df = df.drop(columns=["error"])
        print("🧹 Dropped 'error' column")

    # ---------------------------------------
    # Keep only metric columns that exist
    # ---------------------------------------
    metric_cols_present = [c for c in METRIC_COLS if c in df.columns]

    if not metric_cols_present:
        raise ValueError(f"No valid metric columns found in {csv_path.name}")

    # ---------------------------------------
    # Valid rows = rows with at least one metric
    # ---------------------------------------
    valid_df = df[metric_cols_present].dropna(how="all")

    valid_count = len(valid_df)
    invalid_count = total - valid_count

    # ---------------------------------------
    # Aggregate metrics
    # ---------------------------------------
    avg_metrics = valid_df[metric_cols_present].mean().to_dict()
    std_metrics = valid_df[metric_cols_present].std().to_dict()

    # ---------------------------------------
    # Summary object
    # ---------------------------------------
    summary = {
        "dataset": csv_path.name,
        "total_samples": total,
        "samples_with_metrics": int(valid_count),
        "samples_without_metrics": int(invalid_count),
        "coverage": round(valid_count / total, 4),
        "metrics_mean": avg_metrics,
        "metrics_std": std_metrics,
    }

    summary_file = SUMMARY_DIR / csv_path.name.replace(
        "open_ended_eval_results_", "open_ended_eval_summary_"
    ).replace(".csv", ".json")

    summary_file.write_text(json.dumps(summary, indent=2))

    # ---------------------------------------
    # Console output
    # ---------------------------------------
    print("📊 Dataset Summary")
    print("------------------")
    print(f"Total samples              : {total}")
    print(f"Samples with metrics       : {valid_count}")
    print(f"Samples without metrics    : {invalid_count}")
    print(f"Coverage                   : {valid_count / total:.2%}")

    print("\n📈 Average Metrics")
    for k, v in avg_metrics.items():
        print(f"{k:20s}: {v:.4f}")

    print("\n📉 Std Dev")
    for k, v in std_metrics.items():
        print(f"{k:20s}: {v:.4f}")

    print(f"\n📁 Summary saved to: {summary_file.name}")


# ---------------------------------------
# Run for all CSVs
# ---------------------------------------
print("\n✅ Generating open-ended RAGAS summaries...\n")

for csv_file in csv_files:
    summarize_open_ended(csv_file)

print("\n🎉 All open-ended summaries generated successfully!")
