# run_rag_mcq.py
import pandas as pd
from config import EvalConfig
from rag_pipeline import SimpleRAGPipeline

def load_mcq_dataset(path):
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} MCQ questions")
    return df

def build_mcq_query(row):
    return f"""
Question:
{row['question']}

Options:
A. {row['option_a']}
B. {row['option_b']}
C. {row['option_c']}
D. {row['option_d']}

Answer ONLY with:
- Option letter (A/B/C/D)
- A short explanation
"""

def run_rag_mcq(df, rag):
    answers, contexts = [], []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] MCQ")

        query = build_mcq_query(row)
        ctx = rag.retrieve(query)
        ans = rag.generate(query, ctx)

        answers.append(ans)
        contexts.append(ctx)

    df["model_answer"] = answers
    df["contexts"] = contexts
    return df

if __name__ == "__main__":
    config = EvalConfig()
    rag = SimpleRAGPipeline(config)

    df = load_mcq_dataset(config.MCQ_CSV)
    df = run_rag_mcq(df, rag)

    df.to_csv("rag_outputs_mcq.csv", index=False)
    print("✅ MCQ RAG outputs saved")
