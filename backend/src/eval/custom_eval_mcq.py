# src/eval/custom_eval_mcq.py

import pandas as pd
import re
from pathlib import Path
from src.config import EvalConfig

config = EvalConfig()


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


def evaluate_mcq(df):
    results = []

    for _, row in df.iterrows():
        predicted = extract_option(row["model_answer"])
        correct = str(row["correct_option"]).strip().upper()

        results.append({
            "id": row.get("id"),
            "question": row["question"],
            "predicted_option": predicted,
            "correct_option": correct,
            "is_correct": predicted == correct,
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    input_path = Path(config.OUTPUT_DIR) / "rag_outputs_mcq.csv"
    output_path = Path(config.OUTPUT_DIR) / "mcq_eval_results.csv"

    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} MCQ RAG outputs")

    eval_df = evaluate_mcq(df)
    accuracy = eval_df["is_correct"].mean()

    eval_df.to_csv(output_path, index=False)

    print("✅ MCQ evaluation complete")
    print(f"Overall Accuracy: {accuracy:.3f}")

    print("\nAccuracy by difficulty:")
    print(eval_df.groupby("difficulty")["is_correct"].mean())

    print("\nAccuracy by category:")
    print(eval_df.groupby("category")["is_correct"].mean())
