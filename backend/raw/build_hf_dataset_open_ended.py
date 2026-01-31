# build_hf_dataset_open_ended.py
import pandas as pd
from datasets import Dataset
import ast

# ---------------------------------
# Paths
# ---------------------------------
INPUT_CSV = "rag_outputs_open_ended.csv"
OUTPUT_DIR = "hf_datasets"
DATASET_NAME = "open_ended_eval"

# ---------------------------------
# Load CSV
# ---------------------------------
df = pd.read_csv(INPUT_CSV)

# ---------------------------------
# Convert rows → HF samples
# ---------------------------------
samples = []

for _, row in df.iterrows():
    samples.append({
        "question": row["question"],
        "answer": row["answer"],
        "contexts": ast.literal_eval(row["contexts"]),
        "ground_truth": row["reference"],
    })

# ---------------------------------
# Create HF Dataset
# ---------------------------------
dataset = Dataset.from_list(samples)

# ---------------------------------
# Save to disk
# ---------------------------------
dataset.save_to_disk(f"{OUTPUT_DIR}/{DATASET_NAME}")

print(f"✅ HF Dataset saved at: {OUTPUT_DIR}/{DATASET_NAME}")
print(dataset)
