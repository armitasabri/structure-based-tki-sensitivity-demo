import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "synthetic_engineered.csv" 
OUT_MODELS_DIR = "models"
OUT_RESULTS_DIR = "results"

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

TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_dirs():
    os.makedirs(OUT_MODELS_DIR, exist_ok=True)
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)


def grouped_split(df_drug: pd.DataFrame, test_size=0.2, seed=42):
    """Split using mutation_id as the group to avoid leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X_idx = np.arange(len(df_drug))
    groups = df_drug["mutation_id"].values
    train_idx, test_idx = next(splitter.split(X_idx, groups=groups))
    return train_idx, test_idx


def make_model():
    """
    Simple, robust baseline for demo:
    Logistic Regression + scaling.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])


def eval_binary(y_true, y_pred, y_prob):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def train_one_drug(df_drug: pd.DataFrame, feature_cols, stage_name: str, drug: str):
    #Split
    train_idx, test_idx = grouped_split(df_drug, TEST_SIZE, RANDOM_STATE)
    train_df = df_drug.iloc[train_idx].reset_index(drop=True)
    test_df = df_drug.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    # Train
    model = make_model()
    model.fit(X_train, y_train)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = eval_binary(y_test, y_pred, y_prob)

    # Save model
    model_path = os.path.join(OUT_MODELS_DIR, f"{drug}_{stage_name}.joblib")
    joblib.dump(
        {
            "drug": drug,
            "stage": stage_name,
            "feature_cols": feature_cols,
            "model": model
        },
        model_path
    )

    # Save report
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    out = {
        "drug": drug,
        "stage": stage_name,
        "n_rows": int(len(df_drug)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": metrics,
        "classification_report": report,
        "model_path": model_path,
    }
    return out


def main():
    ensure_dirs()
    df = pd.read_csv(CSV_PATH)

    # Basic checks
    required = {"mutation_id", "drug", "label"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"CSV missing required columns: {missing_req}")

    results = []

    for drug in DRUGS:
        df_drug = df[df["drug"] == drug].copy().reset_index(drop=True)
        if df_drug.empty:
            print(f"[WARN] No rows for {drug}, skipping.")
            continue

        # Stage A: baseline
        res_A = train_one_drug(df_drug, BASELINE_COLS, stage_name="baseline", drug=drug)
        results.append(res_A)

        # Stage B: baseline + engineered + fingerprints
        # ensure all columns exist
        missing_cols = [c for c in FULL_COLS if c not in df_drug.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for FULL stage: {missing_cols[:10]} ... (total {len(missing_cols)})")

        res_B = train_one_drug(df_drug, FULL_COLS, stage_name="full", drug=drug)
        results.append(res_B)

        print(f"\n=== {drug} ===")
        print("Baseline F1:", round(res_A["metrics"]["f1"], 3), "| Full F1:", round(res_B["metrics"]["f1"], 3))

    # Save all results
    out_path = os.path.join(OUT_RESULTS_DIR, "per_tki_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to:", out_path)
    print("Saved models to:", OUT_MODELS_DIR)


if __name__ == "__main__":
    main()
