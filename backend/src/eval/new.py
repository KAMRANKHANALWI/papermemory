# import pandas as pd

# df = pd.read_csv("./Eval_Results/consolidated_eval_results.csv")


# df_clean = df.iloc[:592]

# df_clean = df_clean.reset_index(drop=True)

# df_clean.to_csv("clean_file.csv", index=False)

import pandas as pd

df = pd.read_csv("clean_file.csv")

# Split based on answer_correctness ranges
df_0_to_01 = df[(df["answer_correctness"] >= 0.0) & (df["answer_correctness"] < 0.1)]
df_01_to_02 = df[(df["answer_correctness"] >= 0.1) & (df["answer_correctness"] < 0.2)]
df_02_to_03 = df[(df["answer_correctness"] >= 0.2) & (df["answer_correctness"] < 0.3)]

# Save each as separate CSV
df_0_to_01.to_csv("correctness_0.0_to_0.1.csv", index=False)
df_01_to_02.to_csv("correctness_0.1_to_0.2.csv", index=False)
df_02_to_03.to_csv("correctness_0.2_to_0.3.csv", index=False)

# Quick summary to see how many rows in each
print(f"0.0 - 0.1 : {len(df_0_to_01)} rows")
print(f"0.1 - 0.2 : {len(df_01_to_02)} rows")
print(f"0.2 - 0.3 : {len(df_02_to_03)} rows")