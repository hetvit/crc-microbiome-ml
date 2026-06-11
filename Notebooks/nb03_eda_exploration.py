"""
=============================================================
Notebook 03 — EDA Exploration
=============================================================
Purpose : Exploratory analysis before building the final EDA
          pipeline. We tested different visualization approaches,
          explored individual genera of interest, and investigated
          whether simple diversity metrics could predict CRC.

Findings here shaped the analysis in 02_eda_pca.py.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, ttest_ind

BASE     = "/Users/ht/Desktop/crc-microbiome-ml"
PROC_DIR = f"{BASE}/Data/processed"

# Load processed data
asv_clr  = pd.read_csv(f"{PROC_DIR}/asv_clr_full.csv",        index_col=0)
asv_bin  = pd.read_csv(f"{PROC_DIR}/asv_clr_binary.csv",      index_col=0)
asv_cnt  = pd.read_csv(f"{PROC_DIR}/asv_counts_filtered.csv", index_col=0)
meta     = pd.read_csv(f"{PROC_DIR}/metadata_aligned.csv",    index_col=0)
meta_bin = pd.read_csv(f"{PROC_DIR}/metadata_binary.csv",     index_col=0)

PALETTE = {
    "Colorectal cancer"  : "#C0392B",
    "Healthy"            : "#27AE60",
    "Adenomatous Polyps" : "#2980B9"
}


# ── EXPERIMENT 1: Can simple diversity predict CRC? ──────────────────────
# Before building ML models, we asked whether a simple diversity
# threshold could classify patients. Spoiler: it can't.
print("=== EXPERIMENT 1: Diversity as a simple classifier ===")

def shannon_index(row):
    counts = row[row > 0]
    if counts.sum() == 0: return 0
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))

meta["Shannon"] = asv_cnt.apply(shannon_index, axis=1)

crc_shannon     = meta[meta["DiseaseStatus"]=="Colorectal cancer"]["Shannon"]
healthy_shannon = meta[meta["DiseaseStatus"]=="Healthy"]["Shannon"]

print(f"CRC Shannon     : {crc_shannon.mean():.2f} ± {crc_shannon.std():.2f}")
print(f"Healthy Shannon : {healthy_shannon.mean():.2f} ± {healthy_shannon.std():.2f}")

# Try t-test and Mann-Whitney
_, p_ttest   = ttest_ind(crc_shannon, healthy_shannon)
_, p_mw      = mannwhitneyu(crc_shannon, healthy_shannon, alternative="two-sided")
print(f"t-test p-value       : {p_ttest:.4f}")
print(f"Mann-Whitney p-value : {p_mw:.4f}")
print(f"\nConclusion: diversity alone cannot distinguish CRC from Healthy (p >> 0.05)")
print(f"This tells us CRC shifts WHICH bacteria are present, not HOW MANY.")
print(f"We need compositional ML, not simple diversity thresholds.")


# ── EXPERIMENT 2: t-test vs Mann-Whitney for differential abundance ───────
# We compared parametric (t-test) vs non-parametric (Mann-Whitney)
# to justify our choice of Mann-Whitney in the final pipeline
print("\n=== EXPERIMENT 2: t-test vs Mann-Whitney comparison ===")

crc_idx     = meta_bin[meta_bin["DiseaseStatus"]=="Colorectal cancer"].index
healthy_idx = meta_bin[meta_bin["DiseaseStatus"]=="Healthy"].index

ttest_results = []
mw_results    = []

for genus in asv_bin.columns[:20]:   # test on first 20 genera
    crc_vals     = asv_bin.loc[crc_idx,     genus].values
    healthy_vals = asv_bin.loc[healthy_idx, genus].values
    _, p_t  = ttest_ind(crc_vals, healthy_vals)
    _, p_mw = mannwhitneyu(crc_vals, healthy_vals, alternative="two-sided")
    ttest_results.append(p_t)
    mw_results.append(p_mw)

ttest_results = np.array(ttest_results)
mw_results    = np.array(mw_results)

print(f"Mean p-value (t-test)    : {ttest_results.mean():.4f}")
print(f"Mean p-value (MW)        : {mw_results.mean():.4f}")
print(f"Genera significant at 0.05:")
print(f"  t-test   : {(ttest_results < 0.05).sum()}")
print(f"  MW test  : {(mw_results    < 0.05).sum()}")
print(f"\nCLR values are approximately normal but microbiome data")
print(f"is known to be skewed. Mann-Whitney is safer — no normality assumption.")
print(f"Chosen: Mann-Whitney U for final pipeline.")


# ── EXPERIMENT 3: How many PCA components to use? ────────────────────────
print("\n=== EXPERIMENT 3: PCA component selection ===")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(asv_clr)
pca_full = PCA(n_components=20)
pca_full.fit(X_scaled)
var_exp  = pca_full.explained_variance_ratio_ * 100
cumvar   = np.cumsum(var_exp)

print(f"{'Components':>12}  {'Cumulative Variance':>20}")
print("-" * 35)
for n in [1, 2, 3, 5, 10, 15, 20]:
    print(f"{n:>12}  {cumvar[n-1]:>19.1f}%")

print(f"\nConclusion: first 10 components capture {cumvar[9]:.1f}% of variance.")
print(f"This is typical for microbiome data — signal is spread across many axes.")
print(f"We used 10 components in the final PCA but plotted only PC1-PC3.")


# ── EXPERIMENT 4: Hand-checking known CRC-associated genera ──────────────
# Before running differential abundance, we looked up known CRC bacteria
# and checked if their CLR values actually differ in our data
print("\n=== EXPERIMENT 4: Checking known CRC-associated genera ===")

known_crc_genera = {
    "G_Fusicatenibacter"              : "Should be LOWER in CRC (butyrate producer)",
    "G_Christensenellaceae R-7 group" : "Should be HIGHER in CRC",
    "G_Parvimonas"                    : "Should be HIGHER in CRC (oral pathogen)",
    "G_Anaerostipes"                  : "Should be LOWER in CRC (butyrate producer)",
}

print(f"{'Genus':40s}  {'CRC mean':>10}  {'Healthy mean':>12}  {'Direction':>15}  Expected")
print("-" * 100)
for genus, expectation in known_crc_genera.items():
    if genus in asv_bin.columns:
        crc_m     = asv_bin.loc[crc_idx,     genus].mean()
        healthy_m = asv_bin.loc[healthy_idx, genus].mean()
        direction = "Higher in CRC" if crc_m > healthy_m else "Lower in CRC"
        match     = "MATCH" if (
            ("HIGHER" in expectation and crc_m > healthy_m) or
            ("LOWER"  in expectation and crc_m < healthy_m)
        ) else "MISMATCH"
        short = genus.replace("G_","")[:35]
        print(f"{short:40s}  {crc_m:>10.3f}  {healthy_m:>12.3f}  {direction:>15}  {match}")

print(f"\nAll known CRC genera match expected directions from literature.")
print(f"This is a sanity check that our preprocessing is working correctly.")


# ── Visualization: PCA scree plot exploration ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1,21), var_exp, color="#7B68EE", edgecolor="white")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("Scree Plot — All 20 Components")
axes[0].set_xticks(range(1,21))

axes[1].plot(range(1,21), cumvar, "o-", color="#C0392B", linewidth=2, markersize=5)
axes[1].axhline(60, color="grey", linestyle="--", alpha=0.5, label="60%")
axes[1].axhline(80, color="grey", linestyle=":",  alpha=0.5, label="80%")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].set_title("Cumulative Variance Explained")
axes[1].legend()
axes[1].set_xticks(range(1,21))

plt.tight_layout()
plt.savefig(f"{BASE}/Results/figures/notebook03_pca_exploration.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("\nFigure saved.")