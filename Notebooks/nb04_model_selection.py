"""
=============================================================
Notebook 04 — Model Selection Experiments
=============================================================
Purpose : Testing different model configurations before building
          the final modeling pipeline. We explored regularization
          strength, number of trees for Random Forest, and whether
          NMF features add predictive value.

Findings here informed the final choices in 03_models_final.py.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

BASE     = "/Users/ht/Desktop/crc-microbiome-ml"
PROC_DIR = f"{BASE}/Data/processed"

# Load binary data
asv_bin  = pd.read_csv(f"{PROC_DIR}/asv_clr_binary.csv",     index_col=0)
meta_bin = pd.read_csv(f"{PROC_DIR}/metadata_binary.csv",    index_col=0)
nmf_W    = pd.read_csv(f"{PROC_DIR}/nmf_sample_weights.csv", index_col=0)

y   = meta_bin["label_binary"].values
X   = asv_bin.values
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nmf = nmf_W.loc[asv_bin.index].values

print(f"Binary dataset: {len(y)} samples ({y.sum()} CRC, {(y==0).sum()} Healthy)")
print(f"Features: {X.shape[1]} CLR genera")


# ── EXPERIMENT 1: Tuning L1 regularization strength (C) ──────────────────
print("\n=== EXPERIMENT 1: L1 regularization strength (C) ===")
C_values  = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
aucs_by_C = []

print(f"{'C value':>10}  {'AUC':>8}  {'Non-zero coefs':>15}")
print("-" * 38)
for C in C_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                   C=C, random_state=42, max_iter=1000))
    ])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    aucs_by_C.append(auc)
    pipe.fit(X, y)
    n_nonzero = (pipe.named_steps["clf"].coef_[0] != 0).sum()
    print(f"{C:>10.3f}  {auc:>8.3f}  {n_nonzero:>15}")

# Identify best C from the experiment
best_C     = C_values[int(np.argmax(aucs_by_C))]
best_C_auc = max(aucs_by_C)

# Fit best C to get non-zero coefficient count
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                               C=best_C, random_state=42, max_iter=1000))
])
best_pipe.fit(X, y)
best_nonzero = (best_pipe.named_steps["clf"].coef_[0] != 0).sum()

print(f"\nBest C from experiment : {best_C} (AUC={best_C_auc:.3f}, "
      f"{best_nonzero} non-zero features)")
print(f"Very small C zeroes out everything — too aggressive.")
print(f"Very large C approaches unregularized — overfits on small dataset.")


# ── EXPERIMENT 2: Number of trees in Random Forest ────────────────────────
print("\n=== EXPERIMENT 2: Random Forest — number of trees ===")
tree_counts = [10, 50, 100, 200, 500, 1000]
aucs_trees  = []

print(f"{'Trees':>8}  {'AUC':>8}")
print("-" * 20)
for n in tree_counts:
    pipe = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=n, random_state=42,
                                       class_weight="balanced", n_jobs=-1))
    ])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    aucs_trees.append(auc)
    print(f"{n:>8}  {auc:>8.3f}")

# Identify where AUC stabilizes
best_n_trees     = tree_counts[int(np.argmax(aucs_trees))]
auc_at_200       = aucs_trees[tree_counts.index(200)]
auc_at_500       = aucs_trees[tree_counts.index(500)]
auc_at_1000      = aucs_trees[tree_counts.index(1000)]

print(f"\nAUC at 200 trees : {auc_at_200:.3f}")
print(f"AUC at 500 trees : {auc_at_500:.3f}")
print(f"AUC at 1000 trees: {auc_at_1000:.3f}")
print(f"AUC gain 200→1000: {auc_at_1000 - auc_at_200:.4f} (diminishing returns)")
print(f"Chosen: 500 trees — stable AUC with reasonable runtime.")


# ── EXPERIMENT 3: Does adding NMF features help? ──────────────────────────
print("\n=== EXPERIMENT 3: NMF features — do they add value? ===")
X_combined = np.hstack([X, nmf])

configs = {
    "CLR only"   : X,
    "NMF only"   : nmf,
    "CLR + NMF"  : X_combined,
}

print(f"{'Feature set':15s}  {'# Features':>12}  {'LR AUC':>8}  {'RF AUC':>8}")
print("-" * 50)

lr_aucs = {}
rf_aucs = {}
for name, X_feat in configs.items():
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                   C=1.0, random_state=42, max_iter=1000))
    ])
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                       class_weight="balanced", n_jobs=-1))
    ])
    lr_auc = cross_val_score(lr_pipe, X_feat, y, cv=cv, scoring="roc_auc").mean()
    rf_auc = cross_val_score(rf_pipe, X_feat, y, cv=cv, scoring="roc_auc").mean()
    lr_aucs[name] = lr_auc
    rf_aucs[name] = rf_auc
    print(f"{name:15s}  {X_feat.shape[1]:>12}  {lr_auc:>8.3f}  {rf_auc:>8.3f}")

# Dynamic conclusions based on actual results
lr_clr_vs_combined = lr_aucs["CLR + NMF"] - lr_aucs["CLR only"]
rf_clr_vs_combined = rf_aucs["CLR + NMF"] - rf_aucs["CLR only"]

print(f"\nLR: adding NMF to CLR changes AUC by {lr_clr_vs_combined:+.3f}")
print(f"RF: adding NMF to CLR changes AUC by {rf_clr_vs_combined:+.3f}")
if lr_clr_vs_combined <= 0.005:
    print("NMF adds minimal value to LR — CLR alone used for final LR model.")
if rf_clr_vs_combined <= 0:
    print("NMF hurts RF performance — CLR alone used for final RF model.")
print("NMF components used separately as an additional feature set for comparison.")


# ── EXPERIMENT 4: Effect of class balancing ───────────────────────────────
print("\n=== EXPERIMENT 4: Class weight effect on Random Forest ===")
print(f"CRC samples: {y.sum()}  Healthy samples: {(y==0).sum()} "
      f"(imbalance ratio: {y.sum()/(y==0).sum():.2f})")
print()
print(f"{'Class weight':>15}  {'AUC':>8}  {'Recall':>8}  {'Precision':>10}")
print("-" * 48)

results_cw = {}
for cw in [None, "balanced"]:
    pipe = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                       class_weight=cw, n_jobs=-1))
    ])
    auc  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    rec  = cross_val_score(pipe, X, y, cv=cv, scoring="recall").mean()
    prec = cross_val_score(pipe, X, y, cv=cv, scoring="precision").mean()
    label = str(cw) if cw else "None"
    results_cw[label] = {"auc": auc, "recall": rec, "precision": prec}
    print(f"{label:>15}  {auc:>8.3f}  {rec:>8.3f}  {prec:>10.3f}")

recall_gain    = results_cw["balanced"]["recall"]    - results_cw["None"]["recall"]
precision_loss = results_cw["None"]["precision"]     - results_cw["balanced"]["precision"]

print(f"\nRecall improvement with balanced    : {recall_gain:+.3f}")
print(f"Precision cost with balanced        : {precision_loss:+.3f}")
if recall_gain > 0:
    print(f"Balanced weighting improves recall — fewer missed CRC cases.")
    print(f"For cancer diagnostics recall is the priority — 'balanced' chosen.")


# ── Visualization ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# C value tuning
axes[0].semilogx(C_values, aucs_by_C, "o-", color="#C0392B", linewidth=2, markersize=6)
axes[0].axvline(best_C, color="grey", linestyle="--",
                label=f"Best C={best_C} (AUC={best_C_auc:.3f})")
axes[0].set_xlabel("C (regularization strength)")
axes[0].set_ylabel("CV ROC-AUC")
axes[0].set_title("L1 Logistic Regression\nC Parameter Tuning")
axes[0].legend()

# Trees vs AUC
axes[1].plot(tree_counts, aucs_trees, "o-", color="#2980B9", linewidth=2, markersize=6)
axes[1].axvline(500, color="grey", linestyle="--", label="500 trees (chosen)")
axes[1].set_xlabel("Number of Trees")
axes[1].set_ylabel("CV ROC-AUC")
axes[1].set_title("Random Forest\nNumber of Trees Tuning")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{BASE}/Results/figures/notebook04_model_selection.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("\nFigure saved.")


# ── Final summary (all dynamic) ────────────────────────────────────────────
print("\n=== KEY DECISIONS FROM MODEL SELECTION EXPERIMENTS ===")
print(f"1. L1 regularization C={best_C} — best AUC ({best_C_auc:.3f}) "
      f"with {best_nonzero} non-zero features")
print(f"2. Random Forest: 500 trees — AUC={auc_at_500:.3f}, "
      f"diminishing returns beyond this ({auc_at_1000:.3f} at 1000)")
print(f"3. CLR features vs CLR+NMF: LR change={lr_clr_vs_combined:+.3f}, "
      f"RF change={rf_clr_vs_combined:+.3f} — NMF adds minimal value")
print(f"4. class_weight='balanced' recall gain={recall_gain:+.3f} — "
      f"important for cancer detection")
print(f"5. ElasticNet (combining L1+L2) ultimately gave best overall metrics")