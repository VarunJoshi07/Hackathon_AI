import pandas as pd

df = pd.read_parquet("artifacts/feature_table.parquet")

print(df.columns.tolist())
print(df.iloc[0].to_dict())