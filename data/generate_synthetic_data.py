import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)

N_MUTATIONS = 300
DRUGS = ["Afatinib", "Dacomitinib", "Erlotinib", "Gefitinib", "Osimertinib"]
N_FP = 128  # fingerprint length

rows = []

for m in range(N_MUTATIONS):
    mutation_id = f"MUT_{m:04d}"

    # latent "binding quality" drives correlated features
    z_mut = np.random.normal(0, 1)

    for drug in DRUGS:
        # drug-specific shift (makes drugs behave differently)
        z_drug = np.random.normal(0, 0.4)
        z = z_mut + z_drug

        # --- Baseline features (weak-to-moderate signal)
        rmsd = np.clip(np.random.normal(loc=2.6 - 0.25*z, scale=0.6), 0.5, 6.0)
        affinity = np.random.normal(loc=-9.2 - 0.7*z, scale=0.9)  #note to self: more negative = stronger

        # --- Distance-based (correlated with z)
        min_dist = np.clip(np.random.normal(loc=3.5 - 0.35*z, scale=0.5), 1.5, 8.0)
        mean_contact_dist = np.clip(np.random.normal(loc=5.0 - 0.25*z, scale=0.6), 2.5, 10.0)
        aromatic_min_dist = np.clip(np.random.normal(loc=4.5 - 0.30*z, scale=0.6), 2.0, 10.0)

        # --- Contacts / interactions
        hydrophobic = np.random.poisson(lam=np.clip(6 + 1.2*z, 1, 20))
        hbonds = np.random.poisson(lam=np.clip(2 + 0.6*z, 0, 10))
        contact_richness = np.clip(np.random.normal(loc=12 + 2.0*z, scale=3.0), 0, 40)

        # --- Geometric / shape-based (0..1)
        shape_comp = np.clip(np.random.normal(loc=0.55 + 0.12*z, scale=0.10), 0, 1)
        compactness = np.clip(np.random.normal(loc=0.50 + 0.08*z, scale=0.12), 0, 1)

        # --- Surface area / exposure
        sasa_proxy = np.clip(np.random.normal(loc=420 - 30*z, scale=50), 150, 800)
        buried_fraction = np.clip(np.random.normal(loc=0.45 + 0.10*z, scale=0.12), 0, 1)
        pocket_exposure = np.clip(np.random.normal(loc=0.55 - 0.10*z, scale=0.12), 0, 1)

        # --- Fingerprints: most bits are noise, some bits weakly correlated with z
        # probability shifts slightly with z for "informative" bits
        informative_bits = 24  # how many bits carry signal
        p_base = 0.15
        fp = np.zeros(N_FP, dtype=int)

        # informative block
        p_inf = np.clip(p_base + 0.10 * sigmoid(z), 0.05, 0.35)
        fp[:informative_bits] = (np.random.rand(informative_bits) < p_inf).astype(int)

        # noisy block
        fp[informative_bits:] = (np.random.rand(N_FP - informative_bits) < p_base).astype(int)

        # --- Label generation (uses baseline + engineered + fp signal)
        # NOTE: affinity is negative; stronger binding -> more negative -> higher sensitivity
        # so we use (-affinity) as "strength".
        fp_signal = fp[:informative_bits].sum()

        score = (
            -0.7 * rmsd
            + 0.6 * (-affinity)
            - 0.35 * min_dist
            + 0.25 * hydrophobic
            + 0.30 * hbonds
            + 0.02 * contact_richness
            + 1.2 * shape_comp
            + 0.6 * buried_fraction
            - 0.6 * pocket_exposure
            + 0.12 * fp_signal
        )

        score += np.random.normal(0, 1.0)

        # Convert to probability; threshold yields mildly imbalanced but learnable labels
        p = sigmoid((score - 3.0) / 2.0)
        label = int(np.random.rand() < p)

        row = {
            "mutation_id": mutation_id,
            "drug": drug,
            "rmsd": rmsd,
            "affinity": affinity,
            "min_ligand_protein_distance": min_dist,
            "mean_contact_distance": mean_contact_dist,
            "aromatic_min_distance": aromatic_min_dist,
            "num_hydrophobic_contacts": hydrophobic,
            "num_hbond_contacts": hbonds,
            "contact_richness": contact_richness,
            "shape_complementarity": shape_comp,
            "ligand_compactness": compactness,
            "ligand_sasa_proxy": sasa_proxy,
            "buried_fraction": buried_fraction,
            "pocket_exposure_index": pocket_exposure,
            "label": label,
        }

        # add fp columns
        for i in range(N_FP):
            row[f"fp_{i:03d}"] = int(fp[i])

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("synthetic_engineered.csv", index=False)

print("Saved synthetic_engineered.csv with shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts(normalize=True).round(3))
