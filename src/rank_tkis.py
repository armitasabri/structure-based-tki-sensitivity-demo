import os
import joblib
import numpy as np
import pandas as pd

CSV_PATH = "synthetic_engineered.csv"
MODELS_DIR = "models"
OUT_DIR = "results"

DRUGS = ["Afatinib", "Dacomitinib", "Erlotinib", "Gefitinib", "Osimertinib"]

BASELINE_COLS = ["rmsd", "affinity"]
ENGINEERED_COLS = [
    "min_ligand_protein_distance", "mean_contact_distance", "aromatic_min_distance",
    "num_hydrophobic_contacts", "num_hbond_contacts", "contact_richness",
    "shape_complementarity", "ligand_compactness",
    "ligand_sasa_proxy", "buried_fraction", "pocket_exposure_index",
]
FP_COLS = [f"fp_{i:03d}" for i in range(128)]
FULL_COLS = BASELINE_COLS + ENGINEERED_COLS + FP_COLS


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_full_models():
    models = {}
    for drug in DRUGS:
        path = os.path.join(MODELS_DIR, f"{drug}_full.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        payload = joblib.load(path)
        # payload = {"drug":..., "stage":..., "feature_cols":..., "model": Pipeline}
        models[drug] = payload
    return models


def main():
    ensure_dirs()
    df = pd.read_csv(CSV_PATH)

    # We will rank per mutation_id using the feature row for each (mutation_id, drug).
    required = {"mutation_id", "drug"} | set(FULL_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns (first 10 shown): {sorted(list(missing))[:10]}")

    models = load_full_models()

    ranking_rows = []

    # Compute probabilities for each existing mutation–drug row in the CSV
    for drug in DRUGS:
        df_d = df[df["drug"] == drug].copy()
        if df_d.empty:
            continue

        X = df_d[FULL_COLS].values
        model = models[drug]["model"]
        prob = model.predict_proba(X)[:, 1]

        out = df_d[["mutation_id", "drug"]].copy()
        out["p_sensitive"] = prob
        ranking_rows.append(out)

    rank_df = pd.concat(ranking_rows, ignore_index=True)

    long_path = os.path.join(OUT_DIR, "tki_rankings.csv")
    rank_df.to_csv(long_path, index=False)

    # Sort per mutation by probability descending
    rank_df_sorted = rank_df.sort_values(["mutation_id", "p_sensitive"], ascending=[True, False])

    # For each mutation, collect ordered drugs + probs
    def to_ranked_row(g):
        g = g.sort_values("p_sensitive", ascending=False).reset_index(drop=True)
        row = {"mutation_id": g.loc[0, "mutation_id"]}
        for i in range(len(g)):
            row[f"rank_{i+1}_drug"] = g.loc[i, "drug"]
            row[f"rank_{i+1}_p"] = float(g.loc[i, "p_sensitive"])
        row["best_drug"] = row["rank_1_drug"]
        row["best_p"] = row["rank_1_p"]
        return pd.Series(row)

    best_df = rank_df_sorted.groupby("mutation_id", as_index=False).apply(to_ranked_row)

    wide_path = os.path.join(OUT_DIR, "best_tki_per_mutation.csv")
    best_df.to_csv(wide_path, index=False)

    print("Saved:")
    print(" -", long_path)
    print(" -", wide_path)

    # Quick preview in console
    print("\nPreview of best_tki_per_mutation.csv:")
    print(best_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
