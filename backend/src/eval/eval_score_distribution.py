import pandas as pd

df = pd.read_csv("Eval_Results/consolidated_eval_results.csv")

print(f"Total rows: {len(df)}")

'''
Uncomment below to get the value b/w -> {0.x to 0.y}, i.e., 0.2 to 0.3
'''

# MIN = 0.2
# MAX = 0.3

# filtered = df[
#     (df["answer_correctness"] >= MIN) &
#     (df["answer_correctness"] < MAX)
# ].sort_values("answer_correctness")

# filtered.to_csv(
#     f"Eval_Results/correctness_{MIN}_to_{MAX}.csv",
#     index=False
# )

# print(f"Rows between {MIN} and {MAX}: {len(filtered)}")

MIN=0.2
MAX=1

# split on answer_correctness
low  = df[df["answer_correctness"] <  MIN].sort_values("answer_correctness")
high = df[df["answer_correctness"] >= MIN].sort_values("answer_correctness", ascending=False)

# save
low.to_csv(f"Eval_Results/low_correctness_0_to_{MIN}.csv",   index=False)
high.to_csv(f"Eval_Results/high_correctness_{MIN}_to_1.csv", index=False)

print(f"Low  (0.0 – {MIN}) : {len(low)}  rows  → low_correctness_0_to_{MIN}.csv")
print(f"High ({MIN} – 1.0) : {len(high)} rows  → high_correctness_{MIN}_to_1.csv")

print(df["answer_correctness"].describe())
print(df["answer_correctness"].value_counts(bins=10))

