# custom_eval_mcq.py
import pandas as pd
import re

def extract_option(answer_text: str):
    """
    Extract predicted option letter (A/B/C/D)
    """
    if not isinstance(answer_text, str):
        return None

    match = re.search(r"\b([ABCD])\b", answer_text.upper())
    return match.group(1) if match else None

def evaluate_mcq(df):
    results = []

    for _, row in df.iterrows():
        predicted = extract_option(row["model_answer"])
        correct = row["correct_option"].strip().upper()

        results.append({
            "id": row["id"],
            "question": row["question"],
            "predicted_option": predicted,
            "correct_option": correct,
            "is_correct": predicted == correct,
            "difficulty": row["difficulty"],
            "category": row["category"],
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_csv("rag_outputs_mcq.csv")
    eval_df = evaluate_mcq(df)

    accuracy = eval_df["is_correct"].mean()

    eval_df.to_csv("mcq_eval_results.csv", index=False)

    print("✅ MCQ evaluation complete")
    print(f"Overall Accuracy: {accuracy:.3f}")

    # Optional breakdowns
    print("\nAccuracy by difficulty:")
    print(eval_df.groupby("difficulty")["is_correct"].mean())

    print("\nAccuracy by category:")
    print(eval_df.groupby("category")["is_correct"].mean())
