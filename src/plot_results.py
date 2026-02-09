import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

RANK_LONG = os.path.join(RESULTS_DIR, "tki_rankings.csv")
RANK_WIDE = os.path.join(RESULTS_DIR, "best_tki_per_mutation.csv")
PER_TKI_JSON = os.path.join(RESULTS_DIR, "per_tki_results.json")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved plot:", path)


def plot_best_drug_frequency(best_df: pd.DataFrame):
    counts = best_df["best_drug"].value_counts().sort_values(ascending=False)

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xlabel("Recommended (best) drug")
    plt.ylabel("Number of mutations")
    plt.title("Best TKI frequency across mutations")
    plt.xticks(rotation=30, ha="right")
    savefig("best_drug_frequency.png")


def plot_best_probability_distribution(best_df: pd.DataFrame):
    plt.figure()
    plt.hist(best_df["best_p"].values, bins=25)
    plt.xlabel("Best predicted P(sensitive)")
    plt.ylabel("Count")
    plt.title("Confidence distribution of top recommendation")
    savefig("best_probability_hist.png")


def plot_probability_by_drug(rank_df: pd.DataFrame):
    # Boxplot of predicted probabilities per drug
    drugs = list(rank_df["drug"].unique())
    data = [rank_df.loc[rank_df["drug"] == d, "p_sensitive"].values for d in drugs]

    plt.figure()
    plt.boxplot(data, labels=drugs, showfliers=False)
    plt.xlabel("Drug")
    plt.ylabel("Predicted P(sensitive)")
    plt.title("Predicted sensitivity probability distribution per drug")
    plt.xticks(rotation=30, ha="right")
    savefig("p_sensitive_boxplot_by_drug.png")


def plot_topk_margin(best_df: pd.DataFrame):
    # Margin = best prob - second best prob (how decisive the recommendation is)
    if "rank_2_p" not in best_df.columns:
        print("[WARN] rank_2_p not found; skipping margin plot.")
        return

    margin = best_df["best_p"].astype(float) - best_df["rank_2_p"].astype(float)

    plt.figure()
    plt.hist(margin.values, bins=25)
    plt.xlabel("Margin: best_p _ second_best_p")
    plt.ylabel("Count")
    plt.title("How decisive the top recommendation is")
    savefig("top1_minus_top2_margin_hist.png")


def plot_baseline_vs_full_f1():
    if not os.path.exists(PER_TKI_JSON):
        print("[WARN] per_tki_results.json not found; skipping baseline-vs-full plot.")
        return

    with open(PER_TKI_JSON, "r") as f:
        results = json.load(f)

    # results contains entries for each drug and each stage
    # We'll make a dataframe: rows=drug, columns=stage, metric=f1
    rows = []
    for r in results:
        rows.append({
            "drug": r["drug"],
            "stage": r["stage"],
            "f1": r["metrics"]["f1"]
        })
    dfm = pd.DataFrame(rows)
    pivot = dfm.pivot_table(index="drug", columns="stage", values="f1")

    # Ensure consistent order if possible
    pivot = pivot.sort_index()

    baseline = pivot.get("baseline")
    full = pivot.get("full")
    if baseline is None or full is None:
        print("[WARN] Could not find both baseline and full stages; skipping plot.")
        return

    x = np.arange(len(pivot.index))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, baseline.values, width, label="Baseline (RMSD + affinity)")
    plt.bar(x + width/2, full.values, width, label="Full (engineered + fingerprints)")
    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Drug")
    plt.ylabel("F1-score")
    plt.title("Per-TKI model performance: baseline vs full features")
    plt.legend()
    savefig("baseline_vs_full_f1.png")


def main():
    ensure_dirs()

    rank_df = pd.read_csv(RANK_LONG)
    best_df = pd.read_csv(RANK_WIDE)

    plot_best_drug_frequency(best_df)
    plot_best_probability_distribution(best_df)
    plot_probability_by_drug(rank_df)
    plot_topk_margin(best_df)
    plot_baseline_vs_full_f1()

    print("\nAll plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
