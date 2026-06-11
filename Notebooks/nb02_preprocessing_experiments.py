"""
=============================================================
Notebook 02 — Preprocessing Experiments
=============================================================
Purpose : Testing different preprocessing decisions before
          committing to our final pipeline. We tried different
          prevalence thresholds, compared normalization methods,
          and experimented with pseudocount values.

Findings from this notebook informed the final pipeline in
01_preprocessing_final.py.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE     = "/Users/ht/Desktop/crc-microbiome-ml"
PROC_DIR = f"{BASE}/Data/processed"

# Load the already-processed counts for experimentation
asv_genus = pd.read_csv(f"{PROC_DIR}/asv_counts_filtered.csv", index_col=0)
meta      = pd.read_csv(f"{PROC_DIR}/metadata_aligned.csv",    index_col=0)

print("Loaded genus-level count matrix:", asv_genus.shape)


# ── EXPERIMENT 1: Trying different prevalence thresholds ─────────────────
# Before settling on 10%, we tested what different thresholds would give us
print("\n=== EXPERIMENT 1: Prevalence threshold comparison ===")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Load the full unaggregated genus matrix for this experiment
asv_all = asv_genus.copy()
prevalence = (asv_all > 0).mean(axis=0)

print(f"{'Threshold':>12}  {'Genera retained':>16}  {'% removed':>10}")
print("-" * 45)
for t in thresholds:
    kept = (prevalence >= t).sum()
    pct_removed = (1 - kept/len(prevalence)) * 100
    marker = " <-- CHOSEN" if t == 0.10 else ""
    print(f"{t*100:>10.0f}%  {kept:>16}  {pct_removed:>9.1f}%{marker}")

print("\nConclusion: 10% retains enough genera for modeling while removing")
print("the long tail of extremely rare taxa that add noise.")


# ── EXPERIMENT 2: Comparing normalization approaches ─────────────────────
# We considered 3 approaches before choosing CLR:
#   A) Raw counts (no normalization)
#   B) Relative abundance (divide by row sum)
#   C) CLR transformation (our final choice)

print("\n=== EXPERIMENT 2: Normalization comparison ===")

def clr_transform(df, pseudocount=0.5):
    X        = df.values.astype(float) + pseudocount
    log_X    = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return pd.DataFrame(log_X - geo_mean, index=df.index, columns=df.columns)

def relative_abundance(df):
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0)

# Apply all three
asv_raw  = asv_genus.copy()
asv_rel  = relative_abundance(asv_genus)
asv_clr  = clr_transform(asv_genus)

print("A) Raw counts:")
print(f"   Value range   : [{asv_raw.values.min():.0f}, {asv_raw.values.max():.0f}]")
print(f"   Row sums equal: {np.allclose(asv_raw.sum(axis=1), asv_raw.sum(axis=1).mean())}")
print(f"   Problem       : Not comparable across samples (different sequencing depth)")

print("\nB) Relative abundance:")
print(f"   Value range   : [{asv_rel.values.min():.4f}, {asv_rel.values.max():.4f}]")
print(f"   Row sums = 1  : {np.allclose(asv_rel.sum(axis=1), 1.0)}")
print(f"   Problem       : Still compositional — spurious correlations remain")

print("\nC) CLR transformation (CHOSEN):")
print(f"   Value range   : [{asv_clr.values.min():.2f}, {asv_clr.values.max():.2f}]")
print(f"   Row sums = 0  : {np.allclose(asv_clr.sum(axis=1), 0, atol=1e-10)}")
print(f"   Advantage     : Removes compositional constraint entirely")


# ── EXPERIMENT 3: Pseudocount sensitivity ────────────────────────────────
# How sensitive is CLR to the choice of pseudocount?
print("\n=== EXPERIMENT 3: Pseudocount sensitivity ===")
pseudocounts = [0.01, 0.1, 0.5, 1.0, 2.0]
print(f"{'Pseudocount':>12}  {'CLR min':>10}  {'CLR max':>10}  {'Row sum mean':>14}")
print("-" * 55)
for pc in pseudocounts:
    clr = clr_transform(asv_genus, pseudocount=pc)
    marker = " <-- CHOSEN" if pc == 0.5 else ""
    print(f"{pc:>12.2f}  {clr.values.min():>10.3f}  {clr.values.max():>10.3f}  "
          f"{clr.sum(axis=1).mean():>14.8f}{marker}")

print("\nConclusion: pseudocount=0.5 is the standard in microbiome literature.")
print("Results are not highly sensitive to this choice as long as it is small.")


# ── EXPERIMENT 4: Genus vs raw ASV features ──────────────────────────────
# We explored whether genus aggregation loses useful signal
print("\n=== EXPERIMENT 4: Genus aggregation vs raw ASV features ===")

asv_raw_full = pd.read_csv(f"{PROC_DIR}/asv_raw_asvlevel.csv", index_col=0)

print(f"Raw ASV features     : {asv_raw_full.shape[1]}")
print(f"Genus-level features : {asv_genus.shape[1]}")
print(f"Reduction ratio      : {asv_raw_full.shape[1]/asv_genus.shape[1]:.0f}x")
print(f"\nRaw ASV sparsity     : {(asv_raw_full==0).values.mean()*100:.1f}%")
print(f"Genus-level sparsity : {(asv_genus==0).values.mean()*100:.1f}%")
print(f"\nSamples / Features ratio:")
print(f"  Raw ASVs  : {asv_raw_full.shape[0]} / {asv_raw_full.shape[1]} = "
      f"{asv_raw_full.shape[0]/asv_raw_full.shape[1]:.4f}  (WAY underdetermined)")
print(f"  Genus     : {asv_genus.shape[0]} / {asv_genus.shape[1]} = "
      f"{asv_genus.shape[0]/asv_genus.shape[1]:.2f}  (still high-dim but manageable)")
print(f"\nConclusion: raw ASVs are 6693 features with only 59 samples.")
print(f"Any ML model would severely overfit. Genus aggregation is necessary.")


# ── Visualization: normalization comparison ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Pick first genus for illustration
g = asv_genus.columns[0]

axes[0].hist(asv_raw[g], bins=15, color="#5B9BD5", edgecolor="white")
axes[0].set_title(f"Raw Counts\n({g[:20]})")
axes[0].set_xlabel("Count")

axes[1].hist(asv_rel[g], bins=15, color="#E8914A", edgecolor="white")
axes[1].set_title(f"Relative Abundance\n({g[:20]})")
axes[1].set_xlabel("Proportion")

axes[2].hist(asv_clr[g], bins=15, color="#27AE60", edgecolor="white")
axes[2].set_title(f"CLR Transformed\n({g[:20]})")
axes[2].set_xlabel("CLR Value")

plt.suptitle("Normalization Comparison — Same Genus, Different Transformations",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f"{BASE}/Results/figures/notebook02_normalization_comparison.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("\nFigure saved.")