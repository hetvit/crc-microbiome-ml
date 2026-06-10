"""
=============================================================
CRC Microbiome ML Project — Script 01: Data Preprocessing
=============================================================
Authors : Hetvi Trivedi, Pascal Hayward, Diksha Patel,
          Sarah Quan, Louise Morr, Khaled Al Duwaikat
Course  : UCLA Bioengineering
Script  : 01_preprocessing.py
Purpose : Load raw microbiome sequencing data, aggregate ASVs
          to genus level, filter low-prevalence features, apply
          CLR transformation, and save cleaned data for downstream
          analysis. Also includes a raw ASV baseline comparison
          and confounding variable summary.

Inputs  (from Data/raw/):
    - seqtab_nochim_export.xlsx  : samples x ASV count matrix
    - taxa_species_export.xlsx   : ASV taxonomy reference table
    - metadata.csv               : patient metadata and labels

Outputs (to Data/processed/ and Results/figures/):
    - asv_clr_full.csv           : CLR matrix, all 59 samples
    - asv_clr_binary.csv         : CLR matrix, CRC vs Healthy only
    - asv_counts_filtered.csv    : raw counts post-filtering
    - asv_raw_asvlevel.csv       : raw ASV matrix (no aggregation baseline)
    - metadata_aligned.csv       : aligned metadata, all samples
    - metadata_binary.csv        : metadata, CRC vs Healthy only
    - taxonomy_map.csv           : taxonomy reference with labels
    - 01_preprocessing_qc.png    : QC visualization panel
    - 01_confounding_summary.png : age/sex distribution by group

Usage:
    Run in Google Colab after mounting Google Drive.
    Update BASE path to match your Google Drive folder location.
    Run cells sequentially — this script must be run before
    02_eda_pca.py and 03_models.py.
=============================================================
"""

# ── Standard library ──────────────────────────────────────
import os
import warnings

# ── Third-party ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURATION — update BASE to match your Google Drive path
# =============================================================
BASE      = "/Users/ht/Desktop/crc-microbiome-ml"
RAW_DIR  = f"{BASE}/Data/raw"
PROC_DIR = f"{BASE}/Data/processed"
FIG_DIR  = f"{BASE}/Results/figures"

# Create output directories if they don't exist
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# =============================================================
# FUNCTIONS
# =============================================================

def load_raw_data(raw_dir):
    """
    Load raw ASV table, taxonomy reference, and patient metadata.

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw data files.

    Returns
    -------
    asv_raw : pd.DataFrame
        Samples x ASVs count matrix (59 x 6693).
    taxa_raw : pd.DataFrame
        ASVs x taxonomy levels (6693 x 7).
    meta : pd.DataFrame
        Patient metadata including DiseaseStatus, Age, Sex.
    """
    asv_raw  = pd.read_excel(f"{raw_dir}/seqtab_nochim_export.xlsx",  index_col=0)
    taxa_raw = pd.read_excel(f"{raw_dir}/taxa_species_export.xlsx",   index_col=0)
    meta     = pd.read_csv(  f"{raw_dir}/metadata.csv", sep=";")
    return asv_raw, taxa_raw, meta


def align_samples(asv_raw, meta):
    """
    Align sample IDs between the ASV table and metadata.

    The ASV table uses host_disease codes (e.g. 'CRC1', 'Healthy2')
    as row indices. Metadata is indexed on the same field after
    setting host_disease as the index. Only samples present in
    both tables are retained.

    Parameters
    ----------
    asv_raw : pd.DataFrame
        Raw ASV count matrix with host_disease sample IDs as index.
    meta : pd.DataFrame
        Patient metadata containing a 'host_disease' column.

    Returns
    -------
    asv_raw : pd.DataFrame
        ASV matrix filtered to common samples only.
    meta : pd.DataFrame
        Metadata filtered to common samples only, indexed by host_disease.
    """
    # Set host_disease as index to align with ASV table row labels
    meta = meta.set_index("host_disease")
    common = asv_raw.index.intersection(meta.index)
    asv_raw = asv_raw.loc[common]
    meta    = meta.loc[common]
    return asv_raw, meta


def assign_taxonomy_label(row):
    """
    Assign the most specific available taxonomy label to an ASV row.

    Attempts to assign a genus-level label first. Falls back to
    family, order, class, or phylum if genus is unavailable.
    Labels are prefixed with a single letter indicating the
    taxonomic level (G_, F_, O_, C_, P_) for downstream traceability.

    Parameters
    ----------
    row : pd.Series
        A row from the taxonomy table with columns:
        Kingdom, Phylum, Class, Order, Family, Genus, Species.

    Returns
    -------
    str
        Prefixed taxonomy label (e.g. 'G_Fusicatenibacter') or
        'Unknown' if no valid label is found at any level.
    """
    for level in ["Genus", "Family", "Order", "Class", "Phylum"]:
        val = row.get(level)
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            return f"{level[0]}_{val}"
    return "Unknown"


def aggregate_to_genus(asv_raw, taxa_raw):
    """
    Aggregate ASV-level counts to genus-level features.

    Groups all ASVs sharing the same taxonomy label and sums
    their read counts. This reduces the feature space from
    6,693 ASVs to a smaller set of named genera while retaining
    all count information. No ASVs are dropped at this stage.

    Parameters
    ----------
    asv_raw : pd.DataFrame
        Samples x ASVs raw count matrix.
    taxa_raw : pd.DataFrame
        Taxonomy table with a 'Label' column assigned by
        assign_taxonomy_label().

    Returns
    -------
    asv_genus : pd.DataFrame
        Samples x genera count matrix after aggregation.
    """
    # Transpose so ASVs are rows, then attach taxonomy labels
    asv_t = asv_raw.T.copy()
    asv_t["Label"] = taxa_raw["Label"]

    # Group by label and sum counts, then transpose back to samples x genera
    asv_genus = asv_t.groupby("Label").sum().T
    return asv_genus


def filter_by_prevalence(asv_genus, threshold=0.10):
    """
    Remove genera present in fewer than a given fraction of samples.

    Rare genera (below the prevalence threshold) provide insufficient
    statistical power for differential testing and add noise to ML
    models. The 10% threshold is the standard cutoff in published
    microbiome studies and is applied uniformly across all genera
    with no manual selection.

    Parameters
    ----------
    asv_genus : pd.DataFrame
        Samples x genera count matrix.
    threshold : float
        Minimum fraction of samples a genus must appear in to be
        retained. Default is 0.10 (10%).

    Returns
    -------
    asv_filt : pd.DataFrame
        Filtered samples x genera count matrix.
    prevalence : pd.Series
        Prevalence fraction for each genus (before filtering).
    """
    # Calculate fraction of samples where each genus has count > 0
    prevalence = (asv_genus > 0).mean(axis=0)
    keep_mask  = prevalence >= threshold
    asv_filt   = asv_genus.loc[:, keep_mask]
    return asv_filt, prevalence


def clr_transform(df, pseudocount=0.5):
    """
    Apply Centered Log-Ratio (CLR) transformation to compositional data.

    Microbiome count data is compositional — all values within a
    sample sum to a constant (total sequencing depth), creating
    spurious correlations between features. CLR removes this
    constraint by expressing each feature relative to the geometric
    mean of all features in the same sample.

    Steps:
        1. Add pseudocount to avoid log(0) on zero counts.
        2. Take log of all values.
        3. Subtract row-wise geometric mean (mean of log values).

    After transformation, each row sums to exactly 0, confirming
    the compositional bias has been removed.

    Parameters
    ----------
    df : pd.DataFrame
        Samples x features count matrix (non-negative values).
    pseudocount : float
        Small constant added before log transform to handle zeros.
        Default is 0.5, the standard in microbiome literature.

    Returns
    -------
    pd.DataFrame
        CLR-transformed matrix of same shape as input.
        Row sums should equal 0 (verified in summary output).
    """
    X = df.values.astype(float) + pseudocount  # add pseudocount to avoid log(0)
    log_X    = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)  # row-wise geometric mean in log space
    clr      = log_X - geo_mean                    # subtract geometric mean per sample
    return pd.DataFrame(clr, index=df.index, columns=df.columns)


def build_label_vectors(meta):
    """
    Add binary and 3-class integer label columns to metadata.

    Binary labels are used for the main CRC vs Healthy ML task.
    Adenomatous Polyps are assigned NaN in the binary encoding
    so they can be cleanly excluded from binary analysis while
    remaining available for 3-class exploratory analysis.

    Parameters
    ----------
    meta : pd.DataFrame
        Patient metadata with a 'DiseaseStatus' column containing
        'Colorectal cancer', 'Healthy', or 'Adenomatous Polyps'.

    Returns
    -------
    meta : pd.DataFrame
        Metadata with two new columns:
        - label_binary : int (1=CRC, 0=Healthy, NaN=Polyps)
        - label_3class : int (2=CRC, 1=Polyps, 0=Healthy)
    """
    label_map_binary = {
        "Colorectal cancer"  : 1,
        "Healthy"            : 0,
        "Adenomatous Polyps" : np.nan   # excluded from binary task
    }
    label_map_3class = {
        "Colorectal cancer"  : 2,
        "Healthy"            : 0,
        "Adenomatous Polyps" : 1
    }
    meta = meta.copy()
    meta["label_binary"] = meta["DiseaseStatus"].map(label_map_binary)
    meta["label_3class"] = meta["DiseaseStatus"].map(label_map_3class)
    return meta


def summarize_confounders(meta, fig_dir):
    """
    Generate a summary table and figure of potential confounding
    variables (age and sex) stratified by disease group.

    Age and sex differences between groups could confound ML results.
    This summary allows assessment of whether groups are balanced on
    these variables. If groups differ significantly in age or sex,
    these should be included as covariates in future modeling.

    Parameters
    ----------
    meta : pd.DataFrame
        Patient metadata with columns: DiseaseStatus, Age, Sex.
    fig_dir : str
        Path to save the output figure.

    Returns
    -------
    summary_df : pd.DataFrame
        Table of mean age, age std, and sex counts per group.
    """
    groups = ["Colorectal cancer", "Healthy", "Adenomatous Polyps"]
    palette = {
        "Colorectal cancer"  : "#C0392B",
        "Healthy"            : "#27AE60",
        "Adenomatous Polyps" : "#2980B9"
    }

    # ── Build summary table ───────────────────────────────
    rows = []
    for g in groups:
        sub = meta[meta["DiseaseStatus"] == g]
        rows.append({
            "Group"         : g,
            "N"             : len(sub),
            "Mean Age"      : round(sub["Age"].mean(), 1),
            "Age Std"       : round(sub["Age"].std(), 1),
            "Age Min"       : sub["Age"].min(),
            "Age Max"       : sub["Age"].max(),
            "Male"          : (sub["Sex"] == "male").sum(),
            "Female"        : (sub["Sex"] == "female").sum(),
        })
    summary_df = pd.DataFrame(rows)

    # Chi-square test for sex distribution across groups
    # Tests whether sex composition differs significantly between groups
    sex_table = pd.crosstab(meta["DiseaseStatus"], meta["Sex"])
    chi2, p_sex, _, _ = chi2_contingency(sex_table)

    print("\n  Confounding Variable Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  Sex distribution chi-square test: chi2={chi2:.2f}, p={p_sex:.4f}")
    if p_sex < 0.05:
        print("  ⚠️  Sex distribution significantly differs between groups (p<0.05)")
    else:
        print("  ✅  Sex distribution does not significantly differ between groups")

    # ── Figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # Age distribution by group
    ax0 = axes[0]
    ax0.set_facecolor("#fafafa")
    for g, color in palette.items():
        subset = meta[meta["DiseaseStatus"] == g]["Age"]
        ax0.hist(subset, bins=10, alpha=0.65, label=g,
                 color=color, edgecolor="white")
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.spines["left"].set_color("#cccccc")
    ax0.spines["bottom"].set_color("#cccccc")
    ax0.set_xlabel("Age", fontsize=12, color="#555555")
    ax0.set_ylabel("Count", fontsize=12, color="#555555")
    ax0.set_title("Age Distribution by Disease Group",
                  fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    ax0.legend(fontsize=9)

    # Sex distribution by group (stacked bar)
    ax1 = axes[1]
    ax1.set_facecolor("#fafafa")
    male_counts   = [summary_df[summary_df["Group"]==g]["Male"].values[0]   for g in groups]
    female_counts = [summary_df[summary_df["Group"]==g]["Female"].values[0] for g in groups]
    short_labels  = ["CRC", "Healthy", "Polyps"]
    x = np.arange(len(groups))
    ax1.bar(x, male_counts,   label="Male",   color="#5B9BD5", alpha=0.85, edgecolor="white")
    ax1.bar(x, female_counts, label="Female", color="#E8A0BF", alpha=0.85,
            bottom=male_counts, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=11)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#cccccc")
    ax1.spines["bottom"].set_color("#cccccc")
    ax1.set_ylabel("Count", fontsize=12, color="#555555")
    ax1.set_title(f"Sex Distribution by Disease Group\n(chi-square p={p_sex:.3f})",
                  fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    ax1.legend(fontsize=9)

    fig.suptitle("Confounding Variable Summary — Age & Sex",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/01_confounding_summary.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved: {fig_dir}/01_confounding_summary.png")

    return summary_df


def plot_qc(asv_filt, asv_clr, meta, prevalence, fig_dir):
    """
    Generate a multi-panel QC figure summarizing preprocessing results.

    Panels:
        A) Sample distribution by disease group
        B) Sequencing depth per sample
        C) Sparsity per sample after filtering
        D) Genus prevalence distribution with 10% threshold line
        E) Read depth by disease group (boxplot)
        F) CLR value distribution by group

    Parameters
    ----------
    asv_filt : pd.DataFrame
        Filtered raw count matrix (samples x genera).
    asv_clr : pd.DataFrame
        CLR-transformed matrix (samples x genera).
    meta : pd.DataFrame
        Patient metadata aligned to samples.
    prevalence : pd.Series
        Prevalence fraction per genus before filtering.
    fig_dir : str
        Directory path to save the output figure.

    Returns
    -------
    None
        Saves figure to fig_dir/01_preprocessing_qc.png.
    """
    palette = {
        "Colorectal cancer"  : "#C0392B",
        "Healthy"            : "#27AE60",
        "Adenomatous Polyps" : "#2980B9"
    }

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── A) Sample distribution ────────────────────────────
    ax0    = fig.add_subplot(gs[0, 0])
    counts = meta["DiseaseStatus"].value_counts()
    bars   = ax0.bar(counts.index, counts.values,
                     color=[palette[k] for k in counts.index],
                     edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, counts.values):
        ax0.text(b.get_x() + b.get_width()/2, v + 0.3,
                 str(v), ha="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.set_title("Sample Distribution", fontweight="bold", fontsize=11)
    ax0.set_ylabel("Count")
    ax0.set_xticks(range(len(counts)))
    ax0.set_xticklabels(counts.index, rotation=15, ha="right", fontsize=8)

    # ── B) Sequencing depth ───────────────────────────────
    ax1        = fig.add_subplot(gs[0, 1])
    read_depth = asv_filt.sum(axis=1)
    for status, color in palette.items():
        mask = meta["DiseaseStatus"] == status
        ax1.scatter(range(mask.sum()), sorted(read_depth[mask]),
                    label=status, color=color, alpha=0.85, s=50,
                    edgecolors="white", linewidth=0.5)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.set_title("Sequencing Depth per Sample", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Total Read Count")
    ax1.set_xlabel("Sample (sorted)")
    ax1.legend(fontsize=7)

    # ── C) Sparsity ───────────────────────────────────────
    ax2               = fig.add_subplot(gs[0, 2])
    zeros_per_sample  = (asv_filt == 0).mean(axis=1) * 100
    ax2.hist(zeros_per_sample, bins=15, color="#7B68EE",
             edgecolor="white", linewidth=0.8)
    ax2.axvline(zeros_per_sample.mean(), color="red", linestyle="--",
                label=f"Mean={zeros_per_sample.mean():.1f}%")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.set_title("Sparsity per Sample\n(after filtering)",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("% Zero Entries")
    ax2.set_ylabel("# Samples")
    ax2.legend(fontsize=8)

    # ── D) Genus prevalence ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    # Plot full prevalence distribution including removed genera
    ax3.hist(prevalence * 100, bins=30, color="#E8914A",
             edgecolor="white", linewidth=0.8)
    ax3.axvline(10, color="red", linestyle="--",
                label="10% threshold (filter cutoff)", linewidth=1.5)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    ax3.set_title("Genus Prevalence Distribution",
                  fontweight="bold", fontsize=11)
    ax3.set_xlabel("% Samples Genus Present In")
    ax3.set_ylabel("# Genera")
    ax3.legend(fontsize=8)

    # ── E) Read depth by group ────────────────────────────
    ax4      = fig.add_subplot(gs[1, 1])
    groups   = list(palette.keys())
    box_data = [asv_filt.sum(axis=1)[meta["DiseaseStatus"]==g].values
                for g in groups]
    bp = ax4.boxplot(box_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(palette[g])
        patch.set_alpha(0.75)
    ax4.set_xticks(range(1, len(groups)+1))
    ax4.set_xticklabels(["CRC", "Healthy", "Polyps"],
                        fontsize=9)
    for spine in ["top", "right"]:
        ax4.spines[spine].set_visible(False)
    ax4.set_title("Read Depth by Disease Group",
                  fontweight="bold", fontsize=11)
    ax4.set_ylabel("Total Reads")

    # ── F) CLR value distribution ─────────────────────────
    ax5          = fig.add_subplot(gs[1, 2])
    # Use first 5 genera as representative examples
    sample_genera = asv_clr.columns[:5].tolist()
    clr_long      = asv_clr[sample_genera].copy()
    clr_long["Group"] = meta["DiseaseStatus"].values
    clr_melt = clr_long.melt(id_vars="Group",
                              var_name="Genus", value_name="CLR")
    for status, color in palette.items():
        subset = clr_melt[clr_melt["Group"] == status]["CLR"]
        ax5.hist(subset, bins=25, alpha=0.5, label=status,
                 color=color, edgecolor="none")
    for spine in ["top", "right"]:
        ax5.spines[spine].set_visible(False)
    ax5.set_title("CLR Value Distribution\n(first 5 genera)",
                  fontweight="bold", fontsize=11)
    ax5.set_xlabel("CLR Value")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=7)

    fig.suptitle("Preprocessing QC Report",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(f"{fig_dir}/01_preprocessing_qc.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/01_preprocessing_qc.png")


# =============================================================
# MAIN PIPELINE
# =============================================================

if __name__ == "__main__":

    # ── STEP 1 — Load raw data ────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)

    asv_raw, taxa_raw, meta = load_raw_data(RAW_DIR)

    print(f"  ASV table   : {asv_raw.shape[0]} samples x {asv_raw.shape[1]} ASVs")
    print(f"  Taxonomy    : {taxa_raw.shape[0]} ASVs x {taxa_raw.shape[1]} levels")
    print(f"  Metadata    : {meta.shape[0]} samples x {meta.shape[1]} columns")
    print(f"\n  Disease label distribution:")
    print(meta["DiseaseStatus"].value_counts().to_string(header=False))

    # ── STEP 2 — Align samples ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Aligning sample IDs")
    print("=" * 60)

    asv_raw, meta = align_samples(asv_raw, meta)

    print(f"  Samples retained after alignment: {len(meta)}")

    # ── STEP 3 — Confounding variable summary ─────────────
    print("\n" + "=" * 60)
    print("STEP 3: Confounding variable summary (Age & Sex)")
    print("=" * 60)

    # Important: assess whether age/sex differ across groups
    # before modeling — imbalance could confound predictions
    confounder_summary = summarize_confounders(meta, FIG_DIR)

    # ── STEP 4 — Save raw ASV baseline (no aggregation) ───
    print("\n" + "=" * 60)
    print("STEP 4: Saving raw ASV baseline (no aggregation)")
    print("=" * 60)

    # Save the unaggregated ASV matrix as a baseline reference
    # This allows comparison between genus-aggregated and raw
    # ASV-level features in downstream analysis
    # Note: with 6693 features and only 59 samples this matrix
    # is extremely high-dimensional and will overfit most models
    asv_raw.to_csv(f"{PROC_DIR}/asv_raw_asvlevel.csv")
    print(f"  Saved raw ASV matrix: {asv_raw.shape}")
    print(f"  ⚠️  Note: {asv_raw.shape[1]} features >> {asv_raw.shape[0]} samples")
    print(f"  Raw ASV features are highly sparse: "
          f"{(asv_raw==0).values.mean():.1%} zeros")
    print(f"  Genus aggregation reduces this to a tractable feature space")

    # ── STEP 5 — Assign taxonomy labels ───────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Assigning taxonomy labels")
    print("=" * 60)

    taxa_raw["Label"] = taxa_raw.apply(assign_taxonomy_label, axis=1)

    label_counts = taxa_raw["Label"].str[0].value_counts()
    print(f"  Label resolution breakdown:")
    level_map = {"G": "Genus", "F": "Family", "O": "Order",
                 "C": "Class", "P": "Phylum", "U": "Unknown"}
    for prefix, count in label_counts.items():
        print(f"    {level_map.get(prefix, prefix)}: {count} ASVs")

    # ── STEP 6 — Aggregate to genus level ─────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Aggregating ASVs to genus level")
    print("=" * 60)

    asv_genus = aggregate_to_genus(asv_raw, taxa_raw)

    print(f"  ASVs before aggregation : {asv_raw.shape[1]}")
    print(f"  Genera after aggregation: {asv_genus.shape[1]}")
    print(f"  Reduction               : {asv_raw.shape[1] - asv_genus.shape[1]} features removed")

    # ── STEP 7 — Prevalence filtering ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Prevalence filtering (threshold = 10%)")
    print("=" * 60)

    asv_filt, prevalence = filter_by_prevalence(asv_genus, threshold=0.10)

    print(f"  Genera before filtering : {asv_genus.shape[1]}")
    print(f"  Genera after  filtering : {asv_filt.shape[1]}")
    print(f"  Removed                 : {asv_genus.shape[1] - asv_filt.shape[1]}")
    print(f"  Threshold justification : standard cutoff in microbiome literature")
    print(f"  Statistical basis       : genera in <10% of samples lack power")
    print(f"                            for reliable differential testing")

    # ── STEP 8 — CLR transformation ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: CLR Transformation")
    print("=" * 60)

    asv_clr = clr_transform(asv_filt, pseudocount=0.5)

    # Verify CLR correctness — row sums should equal 0
    row_sum_mean = asv_clr.sum(axis=1).mean()
    print(f"  CLR matrix shape        : {asv_clr.shape}")
    print(f"  CLR value range         : [{asv_clr.values.min():.2f}, {asv_clr.values.max():.2f}]")
    print(f"  Row sums (should be ~0) : mean = {row_sum_mean:.8f}")
    assert abs(row_sum_mean) < 1e-6, "CLR row sums are not zero — transformation failed"
    print(f"  ✅ CLR validation passed")

    # ── STEP 9 — Build label vectors ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Building label vectors")
    print("=" * 60)

    meta = build_label_vectors(meta)

    # Create binary subset by dropping Adenomatous Polyps rows
    binary_mask = meta["label_binary"].notna()
    asv_binary  = asv_clr.loc[binary_mask]
    meta_binary = meta.loc[binary_mask].copy()
    meta_binary["label_binary"] = meta_binary["label_binary"].astype(int)

    print(f"  Full dataset (3-class)  : {asv_clr.shape[0]} samples")
    print(f"  Binary dataset          : {asv_binary.shape[0]} samples")
    print(f"    CRC                   : {(meta_binary['label_binary']==1).sum()}")
    print(f"    Healthy               : {(meta_binary['label_binary']==0).sum()}")
    print(f"  Adenomatous Polyps      : excluded from binary task")
    print(f"    Reason                : only 19 samples per class — too small")
    print(f"    for reliable 3-class ML. Included in exploratory PCA.")
    print(f"    Future direction      : multi-class logistic regression")

    # ── STEP 10 — Save processed data ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: Saving processed data")
    print("=" * 60)

    asv_clr.to_csv(    f"{PROC_DIR}/asv_clr_full.csv")
    meta.to_csv(       f"{PROC_DIR}/metadata_aligned.csv")
    asv_binary.to_csv( f"{PROC_DIR}/asv_clr_binary.csv")
    meta_binary.to_csv(f"{PROC_DIR}/metadata_binary.csv")
    asv_filt.to_csv(   f"{PROC_DIR}/asv_counts_filtered.csv")
    taxa_raw[["Phylum","Class","Order","Family",
              "Genus","Species","Label"]].to_csv(f"{PROC_DIR}/taxonomy_map.csv")
    confounder_summary.to_csv(f"{PROC_DIR}/confounder_summary.csv", index=False)

    print(f"  asv_clr_full.csv        → {asv_clr.shape} (CLR, all 3 classes)")
    print(f"  asv_clr_binary.csv      → {asv_binary.shape} (CLR, CRC vs Healthy)")
    print(f"  asv_counts_filtered.csv → {asv_filt.shape} (raw counts, filtered)")
    print(f"  asv_raw_asvlevel.csv    → {asv_raw.shape} (raw ASV baseline)")
    print(f"  metadata_aligned.csv, metadata_binary.csv")
    print(f"  taxonomy_map.csv, confounder_summary.csv")

    # ── STEP 11 — QC figures ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 11: Generating QC figures")
    print("=" * 60)

    plot_qc(asv_filt, asv_clr, meta, prevalence, FIG_DIR)

    # ── Final summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Raw ASVs (baseline)      : {asv_raw.shape[1]}")
    print(f"  After genus aggregation  : {asv_genus.shape[1]}")
    print(f"  After prevalence filter  : {asv_filt.shape[1]}")
    print(f"  Transformation           : CLR (pseudocount=0.5)")
    print(f"  Samples — full dataset   : {asv_clr.shape[0]}")
    print(f"  Samples — binary task    : {asv_binary.shape[0]}"
          f" ({(meta_binary['label_binary']==1).sum()} CRC, "
          f"{(meta_binary['label_binary']==0).sum()} Healthy)")
    print("=" * 60)
    print("\n  ✅ Ready for 02_eda_pca.py")