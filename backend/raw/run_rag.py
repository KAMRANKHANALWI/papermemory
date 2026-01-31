# run_rag.py
import pandas as pd
from config import EvalConfig
from rag_pipeline import SimpleRAGPipeline

def load_open_ended_dataset(path):
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} questions")
    return df

def run_rag(df, rag):
    answers, contexts = [], []

    for i, row in df.iterrows():
        query = row["question"]
        print(f"[{i+1}/{len(df)}] {query[:60]}")

        ctx = rag.retrieve(query)
        ans = rag.generate(query, ctx)

        answers.append(ans)
        contexts.append(ctx)

    df["answer"] = answers
    df["contexts"] = contexts
    return df

if __name__ == "__main__":
    config = EvalConfig()
    rag = SimpleRAGPipeline(config)

    df = load_open_ended_dataset(config.OPEN_ENDED_CSV)
    df = run_rag(df, rag)

    df.to_csv("rag_outputs_open_ended.csv", index=False)
    print("✅ RAG outputs saved")


