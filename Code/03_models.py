"""
=============================================================
CRC Microbiome ML Project — Script 03: Machine Learning Models
=============================================================
Purpose : Build and compare binary classifiers (CRC vs Healthy)
          using cross-validated evaluation. Includes unregularized
          baseline, logistic regression variants (L1/L2/ElasticNet),
          and Random Forest. Extracts feature importance and
          identifies consensus microbial biomarkers. Assesses
          potential confounding by age and sex.

Inputs  (from Data/processed/):
    - asv_clr_binary.csv         : CLR matrix, CRC vs Healthy
    - metadata_binary.csv        : metadata, CRC vs Healthy
    - nmf_sample_weights.csv     : NMF component weights
    - differential_abundance.csv : differential abundance results

Outputs (to Results/):
    - figures/03_roc_curves.png          : ROC curve comparison
    - figures/03_model_comparison.png    : AUC bar chart
    - figures/03_feature_importance.png  : LR + RF importance
    - figures/03_confusion_matrices.png  : confusion matrices
    - figures/03_confounding_check.png   : age/sex vs predictions
    - metrics/model_comparison.csv       : all model metrics
    - metrics/lr_feature_importance.csv  : LR coefficients
    - metrics/rf_feature_importance.csv  : RF importance scores
    - metrics/consensus_biomarkers.csv   : overlap LR & RF
    - models/lr_elasticnet.pkl           : best LR model
    - models/rf_tuned.pkl                : best RF model
    - models/scaler.pkl                  : fitted StandardScaler

Usage:
    Run after 02_eda_pca.py. Update BASE path to match
    your local project directory.

    python3 Code/03_models.py

Notes:
    - Stratified 5-fold CV used throughout (n=40 too small
      for held-out test set)
    - Unregularized baseline included per TA feedback to show
      effect of regularization
    - ROC curves plotted for ElasticNet vs baseline comparison
    - Age and sex checked as potential confounders
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
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     GridSearchCV)
from sklearn.metrics import (roc_curve, auc, confusion_matrix)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURATION
# =============================================================
BASE      = "/Users/ht/Desktop/crc-microbiome-ml"
PROC_DIR  = f"{BASE}/Data/processed"
FIG_DIR   = f"{BASE}/Results/figures"
RES_DIR   = f"{BASE}/Results/metrics"
MODEL_DIR = f"{BASE}/Results/models"

for d in [FIG_DIR, RES_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Consistent color palette
PALETTE = {
    "Colorectal cancer" : "#C0392B",
    "Healthy"           : "#27AE60"
}

# Literature-validated consensus biomarkers from prior studies
LITERATURE_NOTES = {
    "Fusicatenibacter"              : "Butyrate producer depleted in CRC (Wirbel et al. 2019)",
    "Anaerostipes"                  : "Butyrate producer depleted in CRC (Wirbel et al. 2019)",
    "Lachnospira"                   : "Butyrate producer depleted in CRC (Wirbel et al. 2019)",
    "Christensenellaceae R-7 group" : "CRC-associated taxon (Zeller et al. 2014)",
    "Parvimonas"                    : "Oral pathogen elevated in CRC (Zeller et al. 2014)",
}

# =============================================================
# FUNCTIONS
# =============================================================

def load_data(proc_dir):
    """
    Load all processed data required for model training.

    Parameters
    ----------
    proc_dir : str
        Path to Data/processed/ directory.

    Returns
    -------
    asv_bin : pd.DataFrame
        CLR-transformed matrix, 40 samples (CRC vs Healthy).
    meta_bin : pd.DataFrame
        Metadata for CRC vs Healthy samples.
    nmf_W : pd.DataFrame
        NMF sample weight matrix (59 samples x 5 components).
    diff_df : pd.DataFrame
        Differential abundance results from script 02.
    y : np.ndarray
        Binary label vector (1=CRC, 0=Healthy).
    """
    asv_bin  = pd.read_csv(f"{proc_dir}/asv_clr_binary.csv",         index_col=0)
    meta_bin = pd.read_csv(f"{proc_dir}/metadata_binary.csv",        index_col=0)
    nmf_W    = pd.read_csv(f"{proc_dir}/nmf_sample_weights.csv",     index_col=0)
    diff_df  = pd.read_csv(f"{proc_dir}/differential_abundance.csv")
    y        = meta_bin["label_binary"].values
    return asv_bin, meta_bin, nmf_W, diff_df, y


def prepare_feature_sets(asv_bin, nmf_W, diff_df):
    """
    Prepare multiple feature sets for model comparison.

    Testing multiple feature sets allows assessment of whether
    using all features is better than a targeted subset, and
    whether NMF community features add predictive value.

    Feature sets:
        A - All 122 CLR genera (full feature space)
        B - Top 20 differentially abundant genera (biologically
            motivated subset from script 02 analysis)
        C - 5 NMF community components (heavily compressed)
        D - CLR + NMF combined (127 features)

    Parameters
    ----------
    asv_bin : pd.DataFrame
        CLR matrix, 40 samples x 122 genera.
    nmf_W : pd.DataFrame
        NMF weights, 59 samples x 5 components.
    diff_df : pd.DataFrame
        Differential abundance results with Genus column.

    Returns
    -------
    dict
        Dictionary mapping feature set name to np.ndarray.
    """
    # Align NMF weights to the 40 binary samples
    nmf_bin = nmf_W.loc[asv_bin.index]

    # Top 20 DA genera — biologically motivated feature reduction
    top20_genera = diff_df.head(20)["Genus"].tolist()
    top20_genera = [g for g in top20_genera if g in asv_bin.columns]

    feature_sets = {
        "All CLR (122)"   : asv_bin.values,
        "Top20 DA (20)"   : asv_bin[top20_genera].values,
        "NMF (5)"         : nmf_bin.values,
        "CLR + NMF (127)" : np.hstack([asv_bin.values, nmf_bin.values]),
    }
    return feature_sets


def build_pipelines():
    """
    Build all model pipelines for comparison.

    Each pipeline includes StandardScaler followed by a classifier.
    Scaling is included inside the pipeline to prevent data leakage —
    the scaler is fit only on training folds during cross-validation,
    never on the test fold.

    Models included:
        - Unregularized LR (baseline) : no penalty, will overfit
          on high-dimensional data — included to demonstrate the
          need for regularization per TA feedback
        - LR L1 (Lasso)    : sparse solution, automatic feature selection
        - LR L2 (Ridge)    : shrinks all coefficients, retains all features
        - LR ElasticNet    : combines L1 + L2, best of both
        - Random Forest    : ensemble of 500 trees, handles non-linearity

    Returns
    -------
    dict
        Dictionary mapping model name to sklearn Pipeline.
    """
    pipelines = {
        # Baseline — no regularization
        # Expected to overfit with 122 features and only 40 samples
        "LR-Baseline (no reg)" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty=None, solver="lbfgs",
                random_state=42, max_iter=1000
            ))
        ]),

        # L1 — Lasso: sets unimportant coefficients to exactly zero
        # Best for high-dimensional data where most features are noise
        "LR-L1" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l1", solver="liblinear",
                C=1.0, random_state=42, max_iter=1000
            ))
        ]),

        # L2 — Ridge: shrinks all coefficients but none to zero
        "LR-L2" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l2", solver="lbfgs",
                C=1.0, random_state=42, max_iter=1000
            ))
        ]),

        # ElasticNet — combines L1 and L2 (l1_ratio=0.5)
        # Balances sparsity and stability — best performer in our results
        "LR-ElasticNet" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=1.0, l1_ratio=0.5,
                random_state=42, max_iter=2000
            ))
        ]),

        # Random Forest — ensemble of 500 trees
        # class_weight="balanced" corrects for slight class imbalance
        # (21 CRC vs 19 Healthy)
        "RF" : Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500, random_state=42,
                class_weight="balanced", n_jobs=-1
            ))
        ]),
    }
    return pipelines


def evaluate_model(name, pipeline, X, y, cv):
    """
    Run stratified cross-validation and return mean metrics.

    Stratified 5-fold CV is used throughout because with only
    40 samples a held-out test set would be too small (8 samples)
    to give reliable performance estimates. Stratification ensures
    each fold maintains the same CRC/Healthy ratio as the full dataset.

    Parameters
    ----------
    name : str
        Model name for labeling results.
    pipeline : sklearn Pipeline
        Model pipeline with scaler and classifier.
    X : np.ndarray
        Feature matrix (samples x features).
    y : np.ndarray
        Binary label vector.
    cv : StratifiedKFold
        Cross-validation splitter.

    Returns
    -------
    dict
        Dictionary with model name and mean ± std for each metric.
    """
    scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    scores  = cross_validate(pipeline, X, y, cv=cv,
                             scoring=scoring, return_train_score=False)
    row = {"Model": name}
    for s in scoring:
        vals        = scores[f"test_{s}"]
        row[s.upper()]       = f"{vals.mean():.3f} ± {vals.std():.3f}"
        row[f"_{s}_mean"]    = vals.mean()   # numeric for sorting/plotting
    return row


def plot_roc_curves(pipelines, X, y, cv, fig_dir):
    """
    Plot mean ROC curves with confidence bands for key models.

    Compares ElasticNet (best model) vs unregularized baseline
    to visually demonstrate the effect of regularization.
    Each curve shows the mean TPR across all 5 CV folds with
    ± 1 std shaded band.

    This plot directly addresses TA feedback: "it would be
    helpful to have the ROC curve presented in the write-up,
    so that the difference between the two models is more visible."

    Parameters
    ----------
    pipelines : dict
        Model pipelines — only ElasticNet and baseline are plotted.
    X : np.ndarray
        CLR feature matrix, all 122 genera.
    y : np.ndarray
        Binary label vector.
    cv : StratifiedKFold
        Cross-validation splitter.
    fig_dir : str
        Directory to save the figure.

    Returns
    -------
    None
        Saves figure to fig_dir/03_roc_curves.png.
    """
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1,
            label="Random classifier (AUC=0.5)")

    # Plot only the two most informative models for clarity
    models_to_plot = {
        "LR-ElasticNet"        : ("#C0392B", "ElasticNet LR (best model)"),
        "LR-Baseline (no reg)" : ("#2980B9", "Unregularized LR (baseline)"),
    }

    for model_name, (color, label) in models_to_plot.items():
        pipe     = pipelines[model_name]
        tprs     = []
        aucs     = []
        mean_fpr = np.linspace(0, 1, 100)

        for train, test in cv.split(X, y):
            pipe.fit(X[train], y[train])
            proba = pipe.predict_proba(X[test])[:, 1]
            fpr, tpr, _ = roc_curve(y[test], proba)

            # Interpolate TPR at standard FPR grid for averaging
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
                label=f"{label}\nAUC = {mean_auc:.3f} ± {std_auc:.3f}")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.12, color=color)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.set_xlabel("False Positive Rate", fontsize=12, color="#555555")
    ax.set_ylabel("True Positive Rate", fontsize=12, color="#555555")
    ax.set_title("ROC Curves — ElasticNet vs Unregularized Baseline\n"
                 "(5-fold Cross-Validation, shaded = ±1 std)",
                 fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.7)
    ax.xaxis.grid(True, color="#eeeeee", zorder=0)
    ax.yaxis.grid(True, color="#eeeeee", zorder=0)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_roc_curves.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/03_roc_curves.png")


def plot_model_comparison(results_df, fig_dir):
    """
    Plot horizontal bar chart comparing all models by ROC-AUC.

    Models are sorted by AUC descending. LR models are blue,
    RF models are red. The unregularized baseline is included
    at the top to contextualize the effect of regularization.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model comparison results from evaluate_model().
    fig_dir : str
        Directory to save the figure.

    Returns
    -------
    None
        Saves figure to fig_dir/03_model_comparison.png.
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    plot_df     = results_df.sort_values("_roc_auc_mean", ascending=True)
    bar_colors  = ["#2980B9" if "LR" in m else "#C0392B"
                   for m in plot_df["Model"]]

    bars = ax.barh(range(len(plot_df)), plot_df["_roc_auc_mean"],
                   color=bar_colors, edgecolor="none", alpha=0.85, zorder=3)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Model"], fontsize=9.5, color="#333333")
    ax.invert_yaxis()
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1, zorder=2)
    ax.set_xlim(0.4, 1.08)

    for bar, val in zip(bars, plot_df["_roc_auc_mean"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#333333")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.set_xlabel("ROC-AUC (5-fold Cross-Validation)",
                  fontsize=11, color="#555555")
    ax.set_title("Model Comparison — ROC-AUC",
                 fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)

    legend_els = [
        mpatches.Patch(facecolor="#2980B9", label="Logistic Regression"),
        mpatches.Patch(facecolor="#C0392B", label="Random Forest")
    ]
    ax.legend(handles=legend_els, fontsize=10, framealpha=0.7,
              loc="lower right")
    ax.xaxis.grid(True, color="#eeeeee", zorder=0)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_model_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/03_model_comparison.png")


def plot_feature_importance(lr_importance, rf_imp, overlap, fig_dir):
    """
    Plot LR L1 coefficients and RF feature importance side by side.

    LR coefficients show direction of effect (positive = higher in CRC,
    negative = lower in CRC). RF importance shows mean decrease in
    impurity — a direction-agnostic measure of predictive value.

    Consensus biomarkers (genera in top 20 of both models) are
    the most robust candidates since two fundamentally different
    algorithms agreed on their importance.

    Parameters
    ----------
    lr_importance : pd.DataFrame
        LR feature importance with Genus, Coefficient, Direction columns.
    rf_imp : pd.DataFrame
        RF feature importance with Genus, Importance columns.
    overlap : set
        Set of genera appearing in top 20 of both models.
    fig_dir : str
        Directory to save the figure.

    Returns
    -------
    None
        Saves figure to fig_dir/03_feature_importance.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    # ── LR L1 coefficients ────────────────────────────────
    ax0    = axes[0]
    top15  = lr_importance[lr_importance["Coefficient"] != 0].head(15).copy()
    top15["short"]     = top15["Genus"].str.replace(
        "^G_|^F_|^O_", "", regex=True
    )
    bar_cols = ["#C0392B" if v > 0 else "#27AE60"
                for v in top15["Coefficient"]]
    ax0.set_facecolor("#fafafa")
    ax0.barh(range(len(top15)), top15["Coefficient"].values[::-1],
             color=bar_cols[::-1], edgecolor="none", zorder=3)
    ax0.set_yticks(range(len(top15)))
    ax0.set_yticklabels(top15["short"].values[::-1], fontsize=9.5,
                        color="#333333")
    ax0.axvline(0, color="#333333", linewidth=0.8)
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.set_xlabel("L1 Coefficient", fontsize=11, color="#555555")
    ax0.set_title("Logistic Regression (L1)\nFeature Importance",
                  fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    legend_els = [
        mpatches.Patch(facecolor="#C0392B", label="Higher in CRC"),
        mpatches.Patch(facecolor="#27AE60", label="Higher in Healthy (depleted)")
    ]
    ax0.legend(handles=legend_els, fontsize=9, framealpha=0.5)
    ax0.xaxis.grid(True, color="#eeeeee", zorder=0)

    # ── RF importance ─────────────────────────────────────
    ax1   = axes[1]
    top15_rf = rf_imp.head(15).copy()
    top15_rf["short"] = top15_rf["Genus"].str.replace(
        "^G_|^F_|^O_", "", regex=True
    )
    # Highlight consensus biomarkers in darker red
    rf_colors = ["#922B21" if g in overlap else "#C0392B"
                 for g in top15_rf["Genus"].values[::-1]]
    ax1.set_facecolor("#fafafa")
    ax1.barh(range(15), top15_rf["Importance"].values[::-1],
             color=rf_colors, edgecolor="none", alpha=0.85, zorder=3)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(top15_rf["short"].values[::-1], fontsize=9.5,
                        color="#333333")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.set_xlabel("Mean Decrease in Impurity",
                   fontsize=11, color="#555555")
    ax1.set_title("Random Forest\nFeature Importance (top 15)",
                  fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    legend_els2 = [
        mpatches.Patch(facecolor="#922B21", label="Consensus biomarker (in both LR & RF)"),
        mpatches.Patch(facecolor="#C0392B", label="RF only")
    ]
    ax1.legend(handles=legend_els2, fontsize=9, framealpha=0.5)
    ax1.xaxis.grid(True, color="#eeeeee", zorder=0)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/03_feature_importance.png")


def plot_confusion_matrices(pipelines, X, y, cv, fig_dir):
    """
    Plot aggregated confusion matrices for LR-L1 and RF.

    Confusion matrices are aggregated across all 5 CV folds
    to show the total number of correct and incorrect
    classifications across the full dataset.

    The LR-L1 matrix is the primary clinical result — it shows
    how many CRC cases were correctly caught (true positives)
    and how many were missed (false negatives). Missing a CRC
    case is the most clinically dangerous error type.

    Parameters
    ----------
    pipelines : dict
        Model pipelines — only LR-L1 and RF are shown.
    X : np.ndarray
        CLR feature matrix, all 122 genera.
    y : np.ndarray
        Binary label vector.
    cv : StratifiedKFold
        Cross-validation splitter.
    fig_dir : str
        Directory to save the figure.

    Returns
    -------
    None
        Saves figure to fig_dir/03_confusion_matrices.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")

    for ax, (model_name, label) in zip(axes, [
        ("LR-L1",  "LR-L1 (Lasso)"),
        ("RF",     "Random Forest"),
    ]):
        pipe     = pipelines[model_name]
        cm_total = np.zeros((2, 2), dtype=int)

        # Aggregate confusion matrix across all 5 folds
        for train, test in cv.split(X, y):
            pipe.fit(X[train], y[train])
            preds    = pipe.predict(X[test])
            cm_total += confusion_matrix(y[test], preds)

        sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Healthy", "CRC"],
                    yticklabels=["Healthy", "CRC"],
                    cbar=False, linewidths=1, linecolor="white",
                    annot_kws={"size": 18, "weight": "bold"})
        ax.set_xlabel("Predicted", fontsize=12, color="#555555", labelpad=8)
        ax.set_ylabel("Actual",    fontsize=12, color="#555555", labelpad=8)
        ax.set_title(f"Confusion Matrix — {label}\n(CV aggregated, n=40)",
                     fontsize=12, fontweight="bold", color="#1a1a1a", pad=12)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_confusion_matrices.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/03_confusion_matrices.png")


def check_confounding(meta_bin, asv_bin, y, cv, fig_dir):
    """
    Check whether age and sex are associated with model predictions
    or with disease status in ways that could confound results.

    This addresses the TA feedback: 'How do the differences in age,
    sex, etc. of individuals may or may not affect the results?'

    Checks performed:
        1. Age distribution by disease group (boxplot)
        2. Sex distribution by disease group (bar chart)
        3. Simple logistic regression using age + sex only as features,
           to assess whether demographics alone could predict CRC

    If age/sex alone achieve high AUC, the microbiome signal may
    be partially confounded by demographics. Low AUC from demographics
    alone strengthens confidence in microbiome-driven predictions.

    Parameters
    ----------
    meta_bin : pd.DataFrame
        Metadata for binary task samples with Age and Sex columns.
    asv_bin : pd.DataFrame
        CLR matrix — used only for sample index alignment.
    y : np.ndarray
        Binary label vector.
    cv : StratifiedKFold
        Cross-validation splitter.
    fig_dir : str
        Directory to save the figure.

    Returns
    -------
    confound_auc : float
        Mean CV AUC achieved by demographics-only model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")

    groups  = ["Colorectal cancer", "Healthy"]
    palette = {"Colorectal cancer": "#C0392B", "Healthy": "#27AE60"}

    # ── Age by group ──────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#fafafa")
    box_data = [meta_bin[meta_bin["DiseaseStatus"]==g]["Age"].values
                for g in groups]
    bp = ax0.boxplot(box_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(palette[g])
        patch.set_alpha(0.8)
    for i, (g, color) in enumerate(palette.items()):
        y_pts = meta_bin[meta_bin["DiseaseStatus"]==g]["Age"].values
        x_pts = np.random.normal(i+1, 0.06, size=len(y_pts))
        ax0.scatter(x_pts, y_pts, color=color, alpha=0.6, s=25, zorder=3)
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.set_xticks([1, 2])
    ax0.set_xticklabels(["CRC", "Healthy"], fontsize=11)
    ax0.set_ylabel("Age", fontsize=11, color="#555555")
    ax0.set_title("Age Distribution\nby Disease Group",
                  fontsize=12, fontweight="bold", color="#1a1a1a", pad=10)

    # ── Sex by group ──────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#fafafa")
    for i, (g, color) in enumerate(palette.items()):
        sub    = meta_bin[meta_bin["DiseaseStatus"]==g]
        male   = (sub["Sex"] == "male").sum()
        female = (sub["Sex"] == "female").sum()
        ax1.bar(i - 0.2, male,   0.35, label="Male"   if i==0 else "",
                color="#5B9BD5", alpha=0.85, edgecolor="white")
        ax1.bar(i + 0.2, female, 0.35, label="Female" if i==0 else "",
                color="#E8A0BF", alpha=0.85, edgecolor="white")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["CRC", "Healthy"], fontsize=11)
    ax1.set_ylabel("Count", fontsize=11, color="#555555")
    ax1.set_title("Sex Distribution\nby Disease Group",
                  fontsize=12, fontweight="bold", color="#1a1a1a", pad=10)
    ax1.legend(fontsize=9)

    # ── Demographics-only model ───────────────────────────
    # Build a simple model using only age and sex as features
    # to test whether demographics alone can predict CRC
    ax2 = axes[2]
    ax2.set_facecolor("#fafafa")

    # Encode sex as binary (male=1, female=0)
    sex_encoded = (meta_bin["Sex"] == "male").astype(int).values
    age_vals    = meta_bin["Age"].values
    X_demo      = np.column_stack([age_vals, sex_encoded])

    demo_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l2", solver="lbfgs",
            random_state=42, max_iter=1000
        ))
    ])

    # Get ROC curve for demographics-only model
    tprs_demo = []
    aucs_demo = []
    mean_fpr  = np.linspace(0, 1, 100)

    for train, test in cv.split(X_demo, y):
        demo_pipe.fit(X_demo[train], y[train])
        proba = demo_pipe.predict_proba(X_demo[test])[:, 1]
        fpr, tpr, _ = roc_curve(y[test], proba)
        tprs_demo.append(np.interp(mean_fpr, fpr, tpr))
        aucs_demo.append(auc(fpr, tpr))

    mean_tpr_demo = np.mean(tprs_demo, axis=0)
    confound_auc  = np.mean(aucs_demo)

    ax2.plot([0,1],[0,1],"--", color="grey", linewidth=1,
             label="Random (AUC=0.5)")
    ax2.plot(mean_fpr, mean_tpr_demo, color="#8E44AD", linewidth=2,
             label=f"Age + Sex only\nAUC = {confound_auc:.3f}")
    ax2.fill_between(mean_fpr,
                     mean_tpr_demo - np.std(tprs_demo, axis=0),
                     mean_tpr_demo + np.std(tprs_demo, axis=0),
                     alpha=0.12, color="#8E44AD")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.set_xlabel("False Positive Rate", fontsize=11, color="#555555")
    ax2.set_ylabel("True Positive Rate",  fontsize=11, color="#555555")
    ax2.set_title("Demographics-only Model\n(Age + Sex as features)",
                  fontsize=12, fontweight="bold", color="#1a1a1a", pad=10)
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_confounding_check.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_dir}/03_confounding_check.png")

    return confound_auc


# =============================================================
# MAIN PIPELINE
# =============================================================

if __name__ == "__main__":

    # ── STEP 1 — Load data ────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    asv_bin, meta_bin, nmf_W, diff_df, y = load_data(PROC_DIR)

    print(f"  Samples : {len(y)}  "
          f"(CRC={y.sum()}, Healthy={(y==0).sum()})")
    print(f"  Features (CLR genera) : {asv_bin.shape[1]}")
    print(f"  NMF components        : {nmf_W.shape[1]}")

    # ── STEP 2 — Prepare feature sets ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preparing feature sets")
    print("=" * 60)

    feature_sets = prepare_feature_sets(asv_bin, nmf_W, diff_df)
    for name, X in feature_sets.items():
        print(f"  {name:20s} : {X.shape[1]} features")

    # ── STEP 3 — Build pipelines ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Building model pipelines")
    print("=" * 60)

    pipelines = build_pipelines()
    print(f"  Models to evaluate: {list(pipelines.keys())}")

    # ── STEP 4 — Cross-validation setup ───────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Cross-validation setup")
    print("=" * 60)

    # Stratified 5-fold — preserves CRC/Healthy ratio in each fold
    # With n=40, this is more reliable than a held-out test set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  Method  : Stratified 5-Fold CV")
    print(f"  Samples : {len(y)} total  "
          f"({y.sum()} CRC, {(y==0).sum()} Healthy)")
    print(f"  Rationale: held-out test set too small (n=40)")
    print(f"  Each fold trains on 32 samples, tests on 8")

    # ── STEP 5 — Evaluate all models ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating all models")
    print("=" * 60)

    X_clr = feature_sets["All CLR (122)"]
    results = []

    # Evaluate each LR variant on full CLR features
    for name in ["LR-Baseline (no reg)", "LR-L1", "LR-L2", "LR-ElasticNet"]:
        row = evaluate_model(name, pipelines[name], X_clr, y, cv)
        results.append(row)
        print(f"  {name:25s} | AUC={row['_roc_auc_mean']:.3f} | "
              f"Acc={row['_accuracy_mean']:.3f} | "
              f"Recall={row['_recall_mean']:.3f}")

    # Evaluate RF on all feature sets
    for fname, X in feature_sets.items():
        name = f"RF ({fname})"
        row  = evaluate_model(name, pipelines["RF"], X, y, cv)
        results.append(row)
        print(f"  {name:25s} | AUC={row['_roc_auc_mean']:.3f} | "
              f"Acc={row['_accuracy_mean']:.3f} | "
              f"Recall={row['_recall_mean']:.3f}")

    results_df = pd.DataFrame(results)
    best       = results_df.loc[results_df["_roc_auc_mean"].idxmax()]

    print(f"\n  Best model: {best['Model']} "
          f"(AUC={best['_roc_auc_mean']:.3f})")

    # ── STEP 6 — ROC curve comparison ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: ROC curve comparison (ElasticNet vs Baseline)")
    print("=" * 60)

    plot_roc_curves(pipelines, X_clr, y, cv, FIG_DIR)

    # ── STEP 7 — Feature importance ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Feature importance")
    print("=" * 60)

    # Fit LR-L1 on full data for coefficient extraction
    pipelines["LR-L1"].fit(X_clr, y)
    lr_coefs = pipelines["LR-L1"].named_steps["clf"].coef_[0]

    lr_importance = pd.DataFrame({
        "Genus"       : asv_bin.columns,
        "Coefficient" : lr_coefs
    }).sort_values("Coefficient", key=abs, ascending=False)
    lr_importance["Direction"] = lr_importance["Coefficient"].apply(
        lambda x: "Higher in CRC" if x > 0 else "Lower in CRC"
    )

    # Add literature notes where available
    lr_importance["Literature"] = lr_importance["Genus"].apply(
        lambda g: LITERATURE_NOTES.get(
            g.replace("G_","").replace("F_",""), ""
        )
    )

    nonzero = (lr_importance["Coefficient"] != 0).sum()
    print(f"  LR-L1 non-zero coefficients: {nonzero} / {len(lr_coefs)}")
    print(f"  Top 10 by magnitude:")
    print(lr_importance.head(10)[
        ["Genus", "Coefficient", "Direction", "Literature"]
    ].to_string(index=False))

    # Fit RF on full data for importance extraction
    pipelines["RF"].fit(X_clr, y)
    rf_imp = pd.DataFrame({
        "Genus"      : asv_bin.columns,
        "Importance" : pipelines["RF"].named_steps["clf"].feature_importances_
    }).sort_values("Importance", ascending=False)

    # Find consensus biomarkers — genera in top 20 of BOTH models
    # Agreement between two different algorithms is strong evidence
    lr_top20 = set(lr_importance[
        lr_importance["Coefficient"] != 0
    ].head(20)["Genus"])
    rf_top20 = set(rf_imp.head(20)["Genus"])
    overlap  = lr_top20 & rf_top20

    print(f"\n  Consensus biomarkers (top 20 LR ∩ top 20 RF): "
          f"{len(overlap)}")
    for g in sorted(overlap):
        clean = g.replace("G_","").replace("F_","")
        note  = LITERATURE_NOTES.get(clean, "")
        print(f"    {clean:40s} {note}")

    # ── STEP 8 — Confounding check ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Confounding check (Age & Sex)")
    print("=" * 60)

    confound_auc = check_confounding(
        meta_bin, asv_bin, y, cv, FIG_DIR
    )

    print(f"  Demographics-only AUC: {confound_auc:.3f}")
    print(f"  Microbiome model AUC : {best['_roc_auc_mean']:.3f}")
    print(f"  Difference           : "
          f"{best['_roc_auc_mean'] - confound_auc:.3f}")

    if confound_auc < 0.65:
        print(f"  ✅ Age and sex alone have low predictive power")
        print(f"     Microbiome signal is not primarily driven by")
        print(f"     demographic confounders")
    else:
        print(f"  ⚠️  Age/sex show moderate predictive power")
        print(f"     Future models should include these as covariates")

    # ── STEP 9 — Figures ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Generating figures")
    print("=" * 60)

    plot_model_comparison(results_df, FIG_DIR)
    plot_feature_importance(lr_importance, rf_imp, overlap, FIG_DIR)
    plot_confusion_matrices(pipelines, X_clr, y, cv, FIG_DIR)

    # ── STEP 10 — Save results ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: Saving results")
    print("=" * 60)

    # Model comparison table
    display_cols = ["Model","ACCURACY","ROC_AUC","F1","PRECISION","RECALL"]
    results_df[display_cols].to_csv(
        f"{RES_DIR}/model_comparison.csv", index=False
    )

    # Feature importance files
    lr_importance.to_csv(f"{RES_DIR}/lr_feature_importance.csv", index=False)
    rf_imp.to_csv(       f"{RES_DIR}/rf_feature_importance.csv", index=False)

    # Consensus biomarkers with literature notes
    consensus_df = pd.DataFrame({
        "Genus"      : sorted(overlap),
        "LR_coef"    : [abs(lr_importance[
                            lr_importance["Genus"]==g
                        ]["Coefficient"].values[0]) for g in sorted(overlap)],
        "RF_imp"     : [rf_imp[rf_imp["Genus"]==g]["Importance"].values[0]
                        for g in sorted(overlap)],
        "Direction"  : [lr_importance[
                            lr_importance["Genus"]==g
                        ]["Direction"].values[0] for g in sorted(overlap)],
        "Literature" : [LITERATURE_NOTES.get(
                            g.replace("G_","").replace("F_",""), ""
                        ) for g in sorted(overlap)]
    })
    consensus_df.to_csv(f"{RES_DIR}/consensus_biomarkers.csv", index=False)

    # Save best models
    joblib.dump(pipelines["LR-ElasticNet"], f"{MODEL_DIR}/lr_elasticnet.pkl")
    joblib.dump(pipelines["RF"],            f"{MODEL_DIR}/rf_tuned.pkl")
    scaler = StandardScaler()
    scaler.fit(X_clr)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print(f"  Results/metrics/model_comparison.csv")
    print(f"  Results/metrics/lr_feature_importance.csv")
    print(f"  Results/metrics/rf_feature_importance.csv")
    print(f"  Results/metrics/consensus_biomarkers.csv")
    print(f"  Results/models/lr_elasticnet.pkl")
    print(f"  Results/models/rf_tuned.pkl")
    print(f"  Results/models/scaler.pkl")

    # ── Final summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODELS COMPLETE — Summary")
    print("=" * 60)
    print(f"  Best model           : {best['Model']}")
    print(f"  Best ROC-AUC         : {best['ROC_AUC']}")
    print(f"  Best Accuracy        : {best['ACCURACY']}")
    print(f"  Best F1              : {best['F1']}")
    print(f"  LR non-zero features : {nonzero} / {asv_bin.shape[1]}")
    print(f"  Consensus biomarkers : {len(overlap)}")
    print(f"  Demographics AUC     : {confound_auc:.3f} "
          f"(microbiome = {best['_roc_auc_mean']:.3f})")
    print("=" * 60)
    print("\n  ✅ All outputs saved to Results/")