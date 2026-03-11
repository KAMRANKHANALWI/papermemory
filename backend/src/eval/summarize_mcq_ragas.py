import json
import pandas as pd
from pathlib import Path

# ---------------------------------------
# Paths
# ---------------------------------------
RESULTS_DIR = Path("results")
SUMMARY_DIR = Path("results")

CSV_PATTERN = "mcq_eval_results_*.csv"

csv_files = list(RESULTS_DIR.glob(CSV_PATTERN))

if not csv_files:
    raise FileNotFoundError(
        f"No MCQ result files found matching pattern: {CSV_PATTERN}"
    )

print(f"Found {len(csv_files)} MCQ result file(s):")
for f in csv_files:
    print(" -", f.name)


# ---------------------------------------
# Helper: summarize one CSV
# ---------------------------------------
def summarize_mcq(csv_path: Path):
    df = pd.read_csv(csv_path)

    print(f"\nProcessing: {csv_path.name}")
    print("Detected columns:", df.columns.tolist())

    if "is_correct" not in df.columns:
        raise ValueError(f"'is_correct' column not found in {csv_path.name}")

    # Use ONLY is_correct as ground truth
    valid_df = df[df["is_correct"].isin([True, False])]

    true_count = (valid_df["is_correct"] == True).sum()
    false_count = (valid_df["is_correct"] == False).sum()
    total_attempted = true_count + false_count

    accuracy = (
        round((true_count / total_attempted) * 100, 2) if total_attempted > 0 else 0.0
    )

    skipped = len(df) - total_attempted

    summary = {
        "dataset": csv_path.name,
        "total_rows": int(len(df)),
        "attempted_questions": int(total_attempted),
        "skipped_questions": int(skipped),
        "true_count": int(true_count),
        "false_count": int(false_count),
        "accuracy_percent": float(accuracy),
    }

    summary_file = SUMMARY_DIR / csv_path.name.replace(
        "mcq_eval_results_", "mcq_eval_summary_"
    ).replace(".csv", ".json")

    summary_file.write_text(json.dumps(summary, indent=2))

    # Console output
    print("📊 MCQ Accuracy Summary")
    print("-----------------------")
    print(f"Total rows              : {len(df)}")
    print(f"Attempted questions     : {total_attempted}")
    print(f"Skipped questions       : {skipped}")
    print(f"Correct answers (True)  : {true_count}")
    print(f"Incorrect answers(False): {false_count}")
    print(f"Accuracy (%)            : {accuracy}")
    print(f"📁 Summary saved to     : {summary_file.name}")


# ---------------------------------------
# Run for all CSVs
# ---------------------------------------
print("\n✅ Generating MCQ summaries...\n")

for csv_file in csv_files:
    summarize_mcq(csv_file)

print("\n🎉 All MCQ summaries generated successfully!")
