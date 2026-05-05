import pandas as pd

rag_df  = pd.read_csv("./Eval_Results/rag_outputs_open_ended.csv")
eval_df = pd.read_csv("./Eval_Results/open_ended_eval_results.csv")

print("rag rows:", len(rag_df))
print("eval rows:", len(eval_df))

# Drop duplicate 'question' from eval_df but keep it for the merge key
# Merge on BOTH id + question so each unique question matches correctly
merged_df = pd.merge(rag_df, eval_df, on=["id", "question"], how="left")

print("Merged rows:", len(merged_df))
# Save
merged_df.to_csv("./Eval_Results/consolidated_eval_results.csv", index=False)
print("Done! Shape:", merged_df.shape)
print(merged_df.columns.tolist())

