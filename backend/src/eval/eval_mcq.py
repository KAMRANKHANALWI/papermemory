import json
import pandas as pd
import re
from pathlib import Path
from src.config import EvalConfig

config = EvalConfig()

# ---------------------------------
# Paths
# ---------------------------------
INPUT_CSV = Path(config.OUTPUT_DIR) / "rag_outputs_mcq.csv"
OUTPUT_CSV = Path(config.OUTPUT_DIR) / "mcq_eval_results.csv"
CHECKPOINT = Path("src/eval/checkpoints/eval_mcq_checkpoint.json")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Checkpoint helpers
# ---------------------------------
def load_checkpoint() -> int:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text()).get("last_index", -1)
    return -1


def save_checkpoint(idx: int):
    CHECKPOINT.write_text(json.dumps({"last_index": idx}))


def append_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not OUTPUT_CSV.exists(),
        index=False,
    )

# ---------------------------------
# Option extractor
# ---------------------------------
def extract_option(answer_text: str):
    if not isinstance(answer_text, str):
        return None

    text = answer_text.upper()

    patterns = [
        r"OPTION\s*[:\-]?\s*([ABCD])",
        r"ANSWER\s*[:\-]?\s*([ABCD])",
        r"CORRECT\s+OPTION\s*[:\-]?\s*([ABCD])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None

if __name__ == "__main__":

    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} MCQ RAG outputs")

    start_idx = load_checkpoint()
    print(f"▶️ Resuming from index {start_idx + 1}")

    for idx, row in df.iterrows():

        if idx <= start_idx:
            continue

        try:
            predicted = extract_option(row["model_answer"])
            correct = str(row["correct_option"]).strip().upper()

            result_row = {
                "id": row.get("id", idx + 1),
                "question": row["question"],
                "predicted_option": predicted,
                "correct_option": correct,
                "is_correct": predicted == correct,
                "difficulty": row.get("difficulty"),
                "category": row.get("category"),
            }

            append_row(result_row)
            save_checkpoint(idx)

        except Exception as e:
            print(f"⚠️ Stopped safely at row {idx}: {e}")
            break

    # ---------------------------------
    # Summary 
    # ---------------------------------
    eval_df = pd.read_csv(OUTPUT_CSV)

    print("✅ MCQ evaluation complete")
    print(f"Overall Accuracy: {eval_df['is_correct'].mean():.3f}")

    print("\nAccuracy by difficulty:")
    print(eval_df.groupby("difficulty")["is_correct"].mean())

    print("\nAccuracy by category:")
    print(eval_df.groupby("category")["is_correct"].mean())
