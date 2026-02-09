# Structure-Based TKI Sensitivity Prediction (Public Demo)

## Overview

This repository provides a public, self-contained demo of a structure-based machine learning pipeline for predicting mutation-specific sensitivity to tyrosine kinase inhibitors (TKIs).

The demo is a simplified and reproducible abstraction of a larger Master's thesis project that investigated EGFR mutation–specific drug response using molecular interaction and geometric features derived from protein–ligand complexes.

Due to data confidentiality and proprietary constraints, the original molecular structures and feature-extraction code cannot be released. Instead, this repository demonstrates the core modeling logic, feature engineering strategy, and evaluation workflow using structured synthetic data.

> **Important Note**  
> This repository is intended to demonstrate coding capability and pipeline design, **not** to produce biologically or clinically valid predictions.

## Key Concepts Demonstrated

The demo preserves the methodological structure of the original thesis pipeline:

- Per-mutation, per-drug modeling (one model per TKI)
- Baseline vs expanded feature comparison
- Structured molecular feature categories
- Group-aware train/test splitting to avoid leakage
- Mutation-level ranking of candidate TKIs
- Transparent evaluation and visualization

## Feature Design

Each row in the dataset represents a mutation–drug pair.

### Baseline Features

These reflect the initial descriptors used in the original work:

- **rmsd** — ligand structural deviation
- **affinity** — binding affinity score

### Engineered Feature Categories

These mimic features originally derived from ligand SDF files and protein–ligand complexes:

- **Distance-based features**  
  (e.g., minimum ligand–protein distances, aromatic distances)

- **Contact / interaction features**  
  (e.g., hydrophobic contacts, contact richness)

- **Geometric / shape-based features**  
  (e.g., compactness proxies, shape complementarity)

- **Surface area / exposure features**  
  (e.g., burial and exposure indices)

- **Fingerprint features**  
  Binary molecular fingerprints representing ligand identity

In the original pipeline, these features were computed using **MDAnalysis**, **RDKit**, and **FreeSASA** from standardized ligand SDF files.  
In this demo, they are synthetically generated to reproduce realistic correlations and learnability.

## Repository Structure
```
.
├── data/
│   ├── generate_synthetic_data.py      # Generates structured synthetic dataset
│   └── synthetic_engineered.csv        # Generated dataset
│
├── src/
│   ├── train_per_tki.py                # Per-TKI model training (baseline vs full)
│   ├── rank_tkis.py                    # Mutation-level TKI ranking
│   └── plot_results.py                 # Visualization of results
│
├── models/
│   └── *.joblib                        # Trained per-TKI models
│
├── results/
│   ├── per_tki_results.json            # Evaluation metrics
│   ├── tki_rankings.csv                # Long-format predictions
│   ├── best_tki_per_mutation.csv       # Ranked TKIs per mutation
│   └── plots/                          # Generated figures
│
└── README.md
```

## Pipeline Steps

### 1. Synthetic Data Generation
```bash
python data/generate_synthetic_data.py
```

Generates a structured dataset that mimics the statistical behavior of molecular descriptors used in the original thesis.

### 2. Per-TKI Model Training
```bash
python src/train_per_tki.py
```

- Trains one classifier per TKI
- Compares:
  - Baseline features (RMSD + affinity)
  - Full feature set (engineered + fingerprints)
- Uses group-aware splitting by mutation ID to prevent data leakage

### 3. Mutation-Level TKI Ranking
```bash
python src/rank_tkis.py
```

Applies trained models to rank TKIs for each mutation based on predicted sensitivity probability.

### 4. Visualization
```bash
python src/plot_results.py
```

Produces plots illustrating:

- Performance gains from feature expansion
- Frequency of recommended TKIs
- Confidence margins between top-ranked drugs

## Relationship to the Original Thesis Work

| Original Thesis Pipeline | Public Demo |
|--------------------------|-------------|
| Protein–ligand complexes | Synthetic feature generation |
| Ligand SDF standardization | Abstracted |
| MDAnalysis + RDKit + FreeSASA | Abstracted |
| Batch molecular analysis | Simulated feature output |
| Per-TKI ML models | Fully implemented |
| Drug ranking per mutation | Fully implemented |

The demo should be viewed as a minimal, executable representation of a substantially more complex molecular machine learning pipeline.

## Intended Use

- Evidence of research-level coding ability
- Demonstration of pipeline design and abstraction
- Illustration of feature engineering and evaluation logic
- Supplementary material for PhD or research applications

## Author

**Armita Sabri Kadijani**  
MSc Data Science for Life Sciences  
Structure-based machine learning for drug response modeling
