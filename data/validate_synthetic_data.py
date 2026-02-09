import pandas as pd
import numpy as np

CSV_PATH = "synthetic_engineered.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== Basic info ===")
print("Shape:", df.shape)
print("Columns:", len(df.columns))
print("Missing values total:", int(df.isna().sum().sum()))

print("\n=== Label distribution ===")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))

print("\n=== Per-drug label rate ===")
print(df.groupby("drug")["label"].mean().sort_values(ascending=False).round(3))

print("\n=== Baseline feature stats ===")
print(df[["rmsd", "affinity"]].describe().round(3))

print("\n=== Simple correlations with label (top 15 by abs corr) ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr(numeric_only=True)["label"].drop("label").sort_values(key=lambda s: s.abs(), ascending=False)
print(corr.head(15).round(3))

print("\n=== Fingerprint sanity ===")
fp_cols = [c for c in df.columns if c.startswith("fp_")]
fp_density = df[fp_cols].mean().mean()
print("Fingerprint columns:", len(fp_cols))
print("Average bit density (mean of bits):", round(float(fp_density), 3))

print("\n=== Quick learnability check (baseline rule-of-thumb) ===")
# Not training a model yet; just checking that better affinity + lower rmsd tends to imply label=1 more often
df_tmp = df.copy()
df_tmp["affinity_strength"] = -df_tmp["affinity"]  # more positive = stronger binding
df_tmp["baseline_score"] = 0.6*df_tmp["affinity_strength"] - 0.7*df_tmp["rmsd"]
q = pd.qcut(df_tmp["baseline_score"], 5, duplicates="drop")
print(df_tmp.groupby(q)["label"].mean().round(3))
