"""
=============================================================
Notebook 01 — Initial Data Exploration
=============================================================
Purpose : First look at the raw dataset before any processing.
          Understanding what we are working with — file sizes,
          sparsity, data types, basic summary stats.

This notebook was run before building the preprocessing pipeline
to understand what challenges we would face.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load raw files ────────────────────────────────────────
# NOTE: Update BASE path to your local directory
BASE     = "/Users/ht/Desktop/crc-microbiome-ml"
RAW_DIR  = f"{BASE}/Data/raw"

print("Loading raw data...")
asv  = pd.read_excel(f"{RAW_DIR}/seqtab_nochim_export.xlsx",  index_col=0)
taxa = pd.read_excel(f"{RAW_DIR}/taxa_species_export.xlsx",   index_col=0)
meta = pd.read_csv(  f"{RAW_DIR}/metadata.csv", sep=";")

# ── Basic shape and structure ─────────────────────────────
print("\n=== RAW DATA OVERVIEW ===")
print(f"ASV matrix   : {asv.shape[0]} samples x {asv.shape[1]} ASVs")
print(f"Taxonomy     : {taxa.shape[0]} ASVs x {taxa.shape[1]} levels")
print(f"Metadata     : {meta.shape[0]} samples x {meta.shape[1]} columns")

print("\n--- ASV matrix sample (first 3 rows, first 5 cols) ---")
print(asv.iloc[:3, :5])

print("\n--- Metadata preview ---")
print(meta.head())
print("\nDisease group counts:")
print(meta["DiseaseStatus"].value_counts())

# ── Sparsity check ────────────────────────────────────────
# We expected microbiome data to be sparse — verifying this
sparsity = (asv == 0).values.mean()
print(f"\n=== SPARSITY ===")
print(f"Zero fraction in ASV matrix: {sparsity:.1%}")
print(f"Most bacteria are absent in most samples — very high sparsity")
print(f"This confirms we need prevalence filtering before modeling")

# ── Sequencing depth variability ─────────────────────────
# This was our first signal that normalization would be required
read_depth = asv.sum(axis=1)
print(f"\n=== SEQUENCING DEPTH ===")
print(f"Min reads per sample : {read_depth.min():.0f}")
print(f"Max reads per sample : {read_depth.max():.0f}")
print(f"Mean reads per sample: {read_depth.mean():.0f}")
print(f"Std reads per sample : {read_depth.std():.0f}")
print(f"Depth range ratio    : {read_depth.max()/read_depth.min():.1f}x")
print(f"\nVariability in sequencing depth is ~{read_depth.max()/read_depth.min():.0f}x")
print(f"This means raw counts are NOT comparable across samples")
print(f"Normalization (CLR) will be required")

# ── Taxonomy completeness check ───────────────────────────
# We needed to know how many ASVs had genus-level annotation
print(f"\n=== TAXONOMY COMPLETENESS ===")
for level in ["Kingdom","Phylum","Class","Order","Family","Genus","Species"]:
    pct = taxa[level].notna().mean() * 100
    print(f"  {level:10s}: {pct:.1f}% annotated")

print(f"\nGenus-level annotation is {taxa['Genus'].notna().mean()*100:.1f}%")
print(f"This means some ASVs will need family/order-level fallback labels")

# ── Age and sex distribution ──────────────────────────────
print(f"\n=== CLINICAL METADATA ===")
print(f"Age range: {meta['Age'].min()} - {meta['Age'].max()} years")
print(f"Sex distribution:")
print(meta["Sex"].value_counts().to_string())

print(f"\nAge by disease group:")
for g in meta["DiseaseStatus"].unique():
    sub = meta[meta["DiseaseStatus"]==g]["Age"]
    print(f"  {g}: {sub.mean():.1f} ± {sub.std():.1f} years")

# ── Quick visualization ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Read depth distribution
axes[0].hist(read_depth, bins=20, color="#5B9BD5", edgecolor="white")
axes[0].set_xlabel("Total Reads per Sample")
axes[0].set_ylabel("Count")
axes[0].set_title("Sequencing Depth Distribution\n(raw — before normalization)")

# ASV zero fraction per sample
zeros_per_sample = (asv == 0).mean(axis=1) * 100
axes[1].hist(zeros_per_sample, bins=20, color="#E05C5C", edgecolor="white")
axes[1].set_xlabel("% Zero Entries per Sample")
axes[1].set_ylabel("Count")
axes[1].set_title("Raw Sparsity per Sample\n(before prevalence filtering)")

plt.tight_layout()
plt.savefig(f"{BASE}/Results/figures/notebook01_raw_data_exploration.png",
            dpi=120, bbox_inches="tight")
plt.show()
print(f"\nFigure saved.")

print("\n=== KEY TAKEAWAYS FROM INITIAL EXPLORATION ===")
print(f"1. Data is {sparsity:.1%} sparse — prevalence filtering needed")
print(f"2. Sequencing depth varies ~{read_depth.max()/read_depth.min():.0f}x — CLR normalization needed")
print(f"3. {taxa['Genus'].notna().mean()*100:.0f}% of ASVs annotated at genus level — fallback labels needed")
print(f"4. Disease groups are roughly balanced ({meta['DiseaseStatus'].value_counts().to_dict()})")
print(f"5. Age range {meta['Age'].min()}-{meta['Age'].max()} years, no obvious imbalance across groups")