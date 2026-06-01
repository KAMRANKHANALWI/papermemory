import pandas as pd

# df = pd.read_csv("Eval_Results/consolidated_eval_results.csv")
df = pd.read_csv("Eval_Results/open_ended_eval_results.csv")

print(f"Total rows: {len(df)}")

# split on answer_correctness
# low  = df[df["answer_correctness"] <  0.4].sort_values("answer_correctness")
# high = df[df["answer_correctness"] >= 0.4].sort_values("answer_correctness", ascending=False)
low  = df[df["answer_correctness"] <  0.2].sort_values("answer_correctness")
high = df[df["answer_correctness"] >= 0.2].sort_values("answer_correctness", ascending=False)

# save
low.to_csv("Eval_Results/low_correctness_0_to_0.2.csv",   index=False)
high.to_csv("Eval_Results/high_correctness_0.2_to_1.csv", index=False)

print(f"Low  (0.0 – 0.2) : {len(low)}  rows  → low_correctness_0_to_0.2.csv")
print(f"High (0.2 – 1.0) : {len(high)} rows  → high_correctness_0.2_to_1.csv")

# quick breakdown of the low file by difficulty + category
# print("\n── Low correctness breakdown by difficulty ──")
# print(low["difficulty"].value_counts().to_string())
# print("\n── Low correctness breakdown by category ──")
# print(low["category"].value_counts().to_string())