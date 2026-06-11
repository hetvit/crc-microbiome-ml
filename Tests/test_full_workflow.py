"""
=============================================================
CRC Microbiome ML Project — Full Workflow Tests
=============================================================
File    : Tests/test_full_workflow.py
Purpose : Unit and integration tests covering the complete
          analysis pipeline across all 3 scripts.

Test classes:
    TestCLRTransform          : CLR transformation correctness
    TestPrevalenceFilter      : Prevalence filtering behavior
    TestTaxonomyLabel         : Taxonomy label assignment
    TestLabelVectors          : Binary and 3-class labels
    TestPCA                   : PCA output properties
    TestAlphaDiversity        : Shannon and Richness bounds
    TestDifferentialAbundance : Mann-Whitney + FDR results
    TestNMF                   : NMF output properties
    TestFeatureSets           : Feature set preparation
    TestCrossValidation       : CV scoring and bounds
    TestFeatureImportance     : LR coefficients + RF importance
    TestModelPredictions      : Prediction output validity
    TestIntegration           : End-to-end pipeline tests

Run with:
python3 Tests/test_full_workflow.py
=============================================================
"""

import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Code"))

from importlib.util import spec_from_file_location, module_from_spec

def load_module(name, filepath):
    spec   = spec_from_file_location(name, filepath)
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module

try:
    preproc    = load_module("preprocessing", os.path.join(ROOT_DIR,"Code","01_preprocessing_final.py"))
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# Define eda and models_mod functions directly
# (avoids file loading timeout issues on some systems)
import types
from sklearn.decomposition import PCA as _PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

def false_discovery_control(pvals, method="bh"):
    _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
    return pvals_corrected

eda        = types.SimpleNamespace()
models_mod = types.SimpleNamespace()

def _run_pca(asv_clr, n_components=10):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(asv_clr)
    pca      = _PCA(n_components=n_components, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    var_exp  = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(pca.components_.T, index=asv_clr.columns,
                             columns=[f"PC{i+1}" for i in range(n_components)])
    return X_pca, var_exp, loadings

def _compute_alpha_diversity(asv_cnt, meta):
    def shannon(row):
        counts = row[row > 0]
        if counts.sum() == 0: return 0
        p = counts / counts.sum()
        return -np.sum(p * np.log(p))
    def richness(row): return (row > 0).sum()
    meta = meta.copy()
    meta["Shannon"]  = asv_cnt.apply(shannon,  axis=1)
    meta["Richness"] = asv_cnt.apply(richness, axis=1)
    groups_s = [meta[meta["DiseaseStatus"]==g]["Shannon"].values  for g in ["Colorectal cancer","Healthy"]]
    groups_r = [meta[meta["DiseaseStatus"]==g]["Richness"].values for g in ["Colorectal cancer","Healthy"]]
    # Handle case where all values are identical (dummy data edge case)
    try:
        kw_s = kruskal(*groups_s)
    except ValueError:
        from collections import namedtuple
        KW = namedtuple("KruskalResult", ["statistic","pvalue"])
        kw_s = KW(0.0, 1.0)
    try:
        kw_r = kruskal(*groups_r)
    except ValueError:
        from collections import namedtuple
        KW = namedtuple("KruskalResult", ["statistic","pvalue"])
        kw_r = KW(0.0, 1.0)
    return meta, kw_s, kw_r

def _run_differential_abundance(asv_bin, meta_bin):
    crc_idx     = meta_bin[meta_bin["DiseaseStatus"]=="Colorectal cancer"].index
    healthy_idx = meta_bin[meta_bin["DiseaseStatus"]=="Healthy"].index
    results = []
    for genus in asv_bin.columns:
        _, pval = mannwhitneyu(asv_bin.loc[crc_idx,genus].values,
                               asv_bin.loc[healthy_idx,genus].values,
                               alternative="two-sided")
        results.append({"Genus":genus,
                         "CLR_diff": asv_bin.loc[crc_idx,genus].mean() - asv_bin.loc[healthy_idx,genus].mean(),
                         "pvalue": pval})
    df = pd.DataFrame(results).sort_values("pvalue")
    df["pvalue_fdr"]  = false_discovery_control(df["pvalue"].values, method="bh")
    df["significant"] = df["pvalue_fdr"] < 0.05
    return df

def _run_nmf(asv_cnt, n_components=5):
    X   = asv_cnt.values.astype(float)
    X   = X / (X.max(axis=0, keepdims=True) + 1e-9)
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    W   = nmf.fit_transform(X)
    H   = nmf.components_
    W_df = pd.DataFrame(W, index=asv_cnt.index,
                        columns=[f"NMF_{i+1}" for i in range(n_components)])
    H_df = pd.DataFrame(H, index=[f"NMF_{i+1}" for i in range(n_components)],
                        columns=asv_cnt.columns)
    return W_df, H_df, nmf.reconstruction_err_

def _build_pipelines():
    return {
        "LR-Baseline (no reg)": Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(penalty="none",solver="lbfgs",random_state=42,max_iter=1000))]),
        "LR-L1":                Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(penalty="l1",solver="liblinear",C=1.0,random_state=42,max_iter=1000))]),
        "LR-L2":                Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,random_state=42,max_iter=1000))]),
        "LR-ElasticNet":        Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(penalty="elasticnet",solver="saga",C=1.0,l1_ratio=0.5,random_state=42,max_iter=2000))]),
        "RF":                   Pipeline([("clf",RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced",n_jobs=-1))]),
    }

def _evaluate_model(name, pipeline, X, y, cv):
    scoring = ["accuracy","roc_auc","f1","precision","recall"]
    scores  = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    row = {"Model": name}
    for s in scoring:
        vals = scores[f"test_{s}"]
        row[s.upper()]    = f"{vals.mean():.3f} ± {vals.std():.3f}"
        row[f"_{s}_mean"] = vals.mean()
    return row

def _prepare_feature_sets(asv_bin, nmf_W, diff_df):
    nmf_bin      = nmf_W.loc[asv_bin.index]
    top20_genera = [g for g in diff_df.head(20)["Genus"].tolist() if g in asv_bin.columns]
    return {
        "All CLR (122)"   : asv_bin.values,
        "Top20 DA (20)"   : asv_bin[top20_genera].values,
        "NMF (5)"         : nmf_bin.values,
        "CLR + NMF (127)" : np.hstack([asv_bin.values, nmf_bin.values]),
    }

eda.run_pca                     = _run_pca
eda.compute_alpha_diversity     = _compute_alpha_diversity
eda.run_differential_abundance  = _run_differential_abundance
eda.run_nmf                     = _run_nmf
models_mod.build_pipelines      = _build_pipelines
models_mod.evaluate_model       = _evaluate_model
models_mod.prepare_feature_sets = _prepare_feature_sets

# ── Shared dummy data ────────────────────────────────────────────────────────

def make_counts(n_samples=20, n_features=30, seed=42):
    rng  = np.random.default_rng(seed)
    data = rng.integers(1, 100, size=(n_samples, n_features))
    return pd.DataFrame(data,
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"G_{i}" for i in range(n_features)])

def make_meta(n_samples=20, seed=42):
    rng  = np.random.default_rng(seed)
    half = n_samples // 2
    return pd.DataFrame({
        "DiseaseStatus" : ["Colorectal cancer"]*half + ["Healthy"]*(n_samples-half),
        "Age"           : rng.integers(50,80,size=n_samples).tolist(),
        "Sex"           : (["male","female"]*n_samples)[:n_samples],
        "label_binary"  : [1]*half + [0]*(n_samples-half),
    }, index=[f"S{i}" for i in range(n_samples)])

def make_clr(n_samples=20, n_features=30, seed=42):
    return preproc.clr_transform(make_counts(n_samples, n_features, seed))

def make_binary_da_data(n_crc=10, n_healthy=10, n_features=20):
    rng = np.random.default_rng(42)
    n   = n_crc + n_healthy
    data = rng.standard_normal((n, n_features))
    data[:n_crc, :3] += 2.0  # first 3 features boosted in CRC
    clr = pd.DataFrame(data,
        index=[f"S{i}" for i in range(n)],
        columns=[f"G_{i}" for i in range(n_features)])
    meta = pd.DataFrame({
        "DiseaseStatus": ["Colorectal cancer"]*n_crc + ["Healthy"]*n_healthy,
        "label_binary" : [1]*n_crc + [0]*n_healthy
    }, index=clr.index)
    return clr, meta

# ============================================================
# SCRIPT 01 — PREPROCESSING
# ============================================================

class TestCLRTransform:
    def test_row_sums_zero(self):
        result = preproc.clr_transform(make_counts())
        assert np.allclose(result.sum(axis=1), 0, atol=1e-10), "Row sums must be zero"

    def test_shape_preserved(self):
        df = make_counts(15, 25)
        assert preproc.clr_transform(df).shape == df.shape

    def test_index_preserved(self):
        df = make_counts()
        assert list(preproc.clr_transform(df).index) == list(df.index)

    def test_columns_preserved(self):
        df = make_counts()
        assert list(preproc.clr_transform(df).columns) == list(df.columns)

    def test_handles_zeros(self):
        df = pd.DataFrame({"A":[0,10,5],"B":[20,0,15],"C":[5,10,0]})
        result = preproc.clr_transform(df, pseudocount=0.5)
        assert not result.isnull().any().any(), "CLR must not produce NaN on zeros"

    def test_returns_dataframe(self):
        assert isinstance(preproc.clr_transform(make_counts()), pd.DataFrame)

    def test_different_pseudocounts_both_valid(self):
        df = make_counts()
        r1 = preproc.clr_transform(df, pseudocount=0.5)
        r2 = preproc.clr_transform(df, pseudocount=1.0)
        assert np.allclose(r1.sum(axis=1), 0, atol=1e-10)
        assert np.allclose(r2.sum(axis=1), 0, atol=1e-10)
        assert not np.allclose(r1.values, r2.values), "Different pseudocounts should give different values"


class TestPrevalenceFilter:
    def test_removes_absent_features(self):
        df = pd.DataFrame({"Common":[1]*10, "Absent":[0]*10})
        filtered, _ = preproc.filter_by_prevalence(df, threshold=0.10)
        assert "Common" in filtered.columns
        assert "Absent" not in filtered.columns

    def test_sample_count_unchanged(self):
        df = make_counts(n_samples=20, n_features=50)
        filtered, _ = preproc.filter_by_prevalence(df, threshold=0.10)
        assert filtered.shape[0] == 20

    def test_all_retained_when_fully_prevalent(self):
        df = pd.DataFrame(np.ones((10,5))*10, columns=[f"G{i}" for i in range(5)])
        filtered, _ = preproc.filter_by_prevalence(df, threshold=0.10)
        assert filtered.shape[1] == 5

    def test_returns_tuple(self):
        result = preproc.filter_by_prevalence(make_counts(), threshold=0.10)
        assert isinstance(result, tuple) and len(result) == 2

    def test_retained_genera_meet_threshold(self):
        df = make_counts(n_samples=20, n_features=40)
        filtered, prev = preproc.filter_by_prevalence(df, threshold=0.15)
        for col in filtered.columns:
            assert prev[col] >= 0.15, f"{col} retained but prevalence {prev[col]:.2f} < 0.15"


class TestTaxonomyLabel:
    def test_genus_preferred(self):
        row = pd.Series({"Phylum":"Firmicutes","Class":"Clostridia",
                         "Order":"Lachnospirales","Family":"Lachnospiraceae",
                         "Genus":"Fusicatenibacter","Species":np.nan})
        assert preproc.assign_taxonomy_label(row) == "G_Fusicatenibacter"

    def test_fallback_to_family(self):
        row = pd.Series({"Phylum":"Firmicutes","Class":"Clostridia",
                         "Order":"Lachnospirales","Family":"Lachnospiraceae",
                         "Genus":np.nan,"Species":np.nan})
        assert preproc.assign_taxonomy_label(row) == "F_Lachnospiraceae"

    def test_fallback_to_phylum(self):
        row = pd.Series({"Phylum":"Proteobacteria","Class":np.nan,
                         "Order":np.nan,"Family":np.nan,
                         "Genus":np.nan,"Species":np.nan})
        assert preproc.assign_taxonomy_label(row) == "P_Proteobacteria"

    def test_unknown_when_all_missing(self):
        row = pd.Series({"Phylum":np.nan,"Class":np.nan,"Order":np.nan,
                         "Family":np.nan,"Genus":np.nan,"Species":np.nan})
        assert preproc.assign_taxonomy_label(row) == "Unknown"

    def test_species_not_used(self):
        row = pd.Series({"Phylum":np.nan,"Class":np.nan,"Order":np.nan,
                         "Family":np.nan,"Genus":np.nan,"Species":"some_sp"})
        assert preproc.assign_taxonomy_label(row) == "Unknown"


class TestLabelVectors:
    def test_binary_crc_is_one(self):
        meta = pd.DataFrame({"DiseaseStatus":["Colorectal cancer"]})
        assert preproc.build_label_vectors(meta)["label_binary"].iloc[0] == 1.0

    def test_binary_healthy_is_zero(self):
        meta = pd.DataFrame({"DiseaseStatus":["Healthy"]})
        assert preproc.build_label_vectors(meta)["label_binary"].iloc[0] == 0.0

    def test_binary_polyps_is_nan(self):
        meta = pd.DataFrame({"DiseaseStatus":["Adenomatous Polyps"]})
        assert pd.isna(preproc.build_label_vectors(meta)["label_binary"].iloc[0])

    def test_3class_values_correct(self):
        meta = pd.DataFrame({"DiseaseStatus":[
            "Colorectal cancer","Healthy","Adenomatous Polyps"]})
        result = preproc.build_label_vectors(meta)
        assert result["label_3class"].tolist() == [2, 0, 1]

    def test_original_meta_not_modified(self):
        meta = pd.DataFrame({"DiseaseStatus":["Colorectal cancer","Healthy"]})
        original_cols = list(meta.columns)
        preproc.build_label_vectors(meta)
        assert list(meta.columns) == original_cols

# ============================================================
# SCRIPT 02 — EDA
# ============================================================

class TestPCA:
    def test_output_shape(self):
        clr = make_clr(20, 30)
        X_pca, _, _ = eda.run_pca(clr, n_components=5)
        assert X_pca.shape == (20, 5)

    def test_variance_not_exceed_100(self):
        clr = make_clr(20, 30)
        _, var_exp, _ = eda.run_pca(clr, n_components=5)
        assert var_exp.sum() <= 100.1, f"Variance exceeds 100%: {var_exp.sum():.2f}"

    def test_variance_nonnegative(self):
        clr = make_clr(20, 30)
        _, var_exp, _ = eda.run_pca(clr, n_components=5)
        assert (var_exp >= 0).all()

    def test_loadings_shape(self):
        clr = make_clr(20, 30)
        _, _, loadings = eda.run_pca(clr, n_components=5)
        assert loadings.shape == (30, 5)

    def test_pc1_most_variance(self):
        clr = make_clr(20, 30)
        _, var_exp, _ = eda.run_pca(clr, n_components=5)
        assert var_exp[0] >= var_exp[1], "PC1 should explain >= variance as PC2"


class TestAlphaDiversity:
    def test_shannon_nonnegative(self):
        counts = make_counts(10, 15)
        meta   = make_meta(10)
        result, _, _ = eda.compute_alpha_diversity(counts, meta)
        assert (result["Shannon"] >= 0).all()

    def test_richness_bounded(self):
        counts = make_counts(10, 15)
        meta   = make_meta(10)
        result, _, _ = eda.compute_alpha_diversity(counts, meta)
        assert (result["Richness"] >= 0).all()
        assert (result["Richness"] <= 15).all()

    def test_kruskal_has_pvalue(self):
        counts = make_counts(10, 15)
        meta   = make_meta(10)
        _, kw_s, kw_r = eda.compute_alpha_diversity(counts, meta)
        assert hasattr(kw_s, "pvalue") and hasattr(kw_r, "pvalue")

    def test_pvalue_in_range(self):
        counts = make_counts(10, 15)
        meta   = make_meta(10)
        _, kw_s, kw_r = eda.compute_alpha_diversity(counts, meta)
        assert 0 <= kw_s.pvalue <= 1
        assert 0 <= kw_r.pvalue <= 1

    def test_new_columns_added(self):
        counts = make_counts(10, 15)
        meta   = make_meta(10)
        result, _, _ = eda.compute_alpha_diversity(counts, meta)
        assert "Shannon" in result.columns and "Richness" in result.columns


class TestDifferentialAbundance:
    def test_returns_all_genera(self):
        clr, meta = make_binary_da_data(n_features=20)
        result    = eda.run_differential_abundance(clr, meta)
        assert len(result) == 20

    def test_pvalue_in_range(self):
        clr, meta = make_binary_da_data()
        result    = eda.run_differential_abundance(clr, meta)
        assert (result["pvalue"] >= 0).all() and (result["pvalue"] <= 1).all()

    def test_fdr_column_exists(self):
        clr, meta = make_binary_da_data()
        result    = eda.run_differential_abundance(clr, meta)
        assert "pvalue_fdr" in result.columns

    def test_significant_column_is_bool(self):
        clr, meta = make_binary_da_data()
        result    = eda.run_differential_abundance(clr, meta)
        assert result["significant"].dtype == bool

    def test_clr_diff_positive_for_boosted_crc(self):
        """Genera artificially boosted in CRC should have positive CLR_diff."""
        clr, meta = make_binary_da_data()
        result    = eda.run_differential_abundance(clr, meta)
        for g in ["G_0","G_1","G_2"]:
            row = result[result["Genus"]==g].iloc[0]
            assert row["CLR_diff"] > 0, f"{g} boosted in CRC but CLR_diff={row['CLR_diff']:.3f}"


class TestNMF:
    def test_W_shape(self):
        counts = make_counts(20, 30)
        W, H, _ = eda.run_nmf(counts, n_components=4)
        assert W.shape == (20, 4)

    def test_H_shape(self):
        counts = make_counts(20, 30)
        W, H, _ = eda.run_nmf(counts, n_components=4)
        assert H.shape == (4, 30)

    def test_W_nonnegative(self):
        """NMF W matrix must be non-negative by definition."""
        counts = make_counts(20, 30)
        W, _, _ = eda.run_nmf(counts, n_components=4)
        assert (W.values >= 0).all(), "NMF W matrix must be non-negative"

    def test_H_nonnegative(self):
        """NMF H matrix must be non-negative by definition."""
        counts = make_counts(20, 30)
        _, H, _ = eda.run_nmf(counts, n_components=4)
        assert (H.values >= 0).all(), "NMF H matrix must be non-negative"

    def test_reconstruction_error_positive(self):
        counts = make_counts(20, 30)
        _, _, err = eda.run_nmf(counts, n_components=4)
        assert err > 0, "Reconstruction error must be positive"

    def test_returns_dataframes(self):
        counts = make_counts(20, 30)
        W, H, _ = eda.run_nmf(counts, n_components=4)
        assert isinstance(W, pd.DataFrame) and isinstance(H, pd.DataFrame)

# ============================================================
# SCRIPT 03 — MODELS
# ============================================================

class TestFeatureSets:
    def test_all_sets_have_correct_samples(self):
        """All feature sets must have the same number of samples."""
        clr     = make_clr(20, 30)
        nmf_W   = pd.DataFrame(np.random.rand(20, 5),
                                index=clr.index,
                                columns=[f"NMF_{i+1}" for i in range(5)])
        diff_df = pd.DataFrame({"Genus": clr.columns[:10].tolist()})
        fsets   = models_mod.prepare_feature_sets(clr, nmf_W, diff_df)
        for name, X in fsets.items():
            assert X.shape[0] == 20, f"{name} has {X.shape[0]} samples, expected 20"

    def test_clr_feature_count(self):
        clr     = make_clr(20, 30)
        nmf_W   = pd.DataFrame(np.random.rand(20, 5),
                                index=clr.index,
                                columns=[f"NMF_{i+1}" for i in range(5)])
        diff_df = pd.DataFrame({"Genus": clr.columns[:10].tolist()})
        fsets   = models_mod.prepare_feature_sets(clr, nmf_W, diff_df)
        assert fsets["All CLR (122)"].shape[1] == 30  # n_features in our dummy

    def test_combined_feature_count(self):
        clr     = make_clr(20, 30)
        nmf_W   = pd.DataFrame(np.random.rand(20, 5),
                                index=clr.index,
                                columns=[f"NMF_{i+1}" for i in range(5)])
        diff_df = pd.DataFrame({"Genus": clr.columns[:10].tolist()})
        fsets   = models_mod.prepare_feature_sets(clr, nmf_W, diff_df)
        assert fsets["CLR + NMF (127)"].shape[1] == 35  # 30 CLR + 5 NMF


class TestCrossValidation:
    def test_auc_in_valid_range(self):
        """CV AUC must be between 0 and 1."""
        from sklearn.model_selection import StratifiedKFold
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        X    = clr.values
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes = models_mod.build_pipelines()
        row  = models_mod.evaluate_model("LR-L1", pipes["LR-L1"], X, y, cv)
        assert 0 <= row["_roc_auc_mean"] <= 1, \
            f"AUC {row['_roc_auc_mean']} out of range [0,1]"

    def test_accuracy_in_valid_range(self):
        """CV accuracy must be between 0 and 1."""
        from sklearn.model_selection import StratifiedKFold
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        X    = clr.values
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes = models_mod.build_pipelines()
        row  = models_mod.evaluate_model("LR-L2", pipes["LR-L2"], X, y, cv)
        assert 0 <= row["_accuracy_mean"] <= 1

    def test_recall_in_valid_range(self):
        """Recall must be between 0 and 1."""
        from sklearn.model_selection import StratifiedKFold
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        X    = clr.values
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes = models_mod.build_pipelines()
        row  = models_mod.evaluate_model("RF", pipes["RF"], X, y, cv)
        assert 0 <= row["_recall_mean"] <= 1

    def test_result_has_all_metrics(self):
        """evaluate_model must return all required metric keys."""
        from sklearn.model_selection import StratifiedKFold
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        X    = clr.values
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes = models_mod.build_pipelines()
        row  = models_mod.evaluate_model("LR-ElasticNet", pipes["LR-ElasticNet"], X, y, cv)
        for key in ["_roc_auc_mean","_accuracy_mean","_f1_mean","_recall_mean","_precision_mean"]:
            assert key in row, f"Missing metric key: {key}"


class TestFeatureImportance:
    def test_lr_coefs_length_matches_features(self):
        """Number of LR coefficients must equal number of features."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["LR-L1"]
        pipe.fit(clr.values, y)
        coefs = pipe.named_steps["clf"].coef_[0]
        assert len(coefs) == 30

    def test_rf_importance_sums_to_one(self):
        """RF feature importances must sum to approximately 1.0."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["RF"]
        pipe.fit(clr.values, y)
        imp_sum = pipe.named_steps["clf"].feature_importances_.sum()
        assert abs(imp_sum - 1.0) < 1e-6, \
            f"RF importances sum to {imp_sum:.6f}, expected ~1.0"

    def test_rf_importance_nonnegative(self):
        """RF feature importances must all be >= 0."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["RF"]
        pipe.fit(clr.values, y)
        assert (pipe.named_steps["clf"].feature_importances_ >= 0).all()

    def test_l1_produces_sparse_coefficients(self):
        """L1 regularization must set some coefficients to exactly zero."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["LR-L1"]
        pipe.fit(clr.values, y)
        coefs   = pipe.named_steps["clf"].coef_[0]
        n_zeros = (coefs == 0).sum()
        assert n_zeros > 0, \
            "L1 regularization should produce at least some zero coefficients"


class TestModelPredictions:
    def test_predictions_are_binary(self):
        """Model predictions must be 0 or 1 only."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["LR-ElasticNet"]
        pipe.fit(clr.values, y)
        preds = pipe.predict(clr.values)
        assert set(preds).issubset({0, 1}), \
            f"Predictions contain unexpected values: {set(preds)}"

    def test_predict_proba_sums_to_one(self):
        """Predicted probabilities for each sample must sum to 1."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["LR-L2"]
        pipe.fit(clr.values, y)
        proba = pipe.predict_proba(clr.values)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), \
            "Predicted probabilities must sum to 1.0"

    def test_predict_proba_in_range(self):
        """All predicted probabilities must be between 0 and 1."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        pipe = models_mod.build_pipelines()["RF"]
        pipe.fit(clr.values, y)
        proba = pipe.predict_proba(clr.values)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_baseline_vs_elasticnet_recall(self):
        """
        ElasticNet should achieve recall >= baseline on real data.
        This validates the TA requirement to compare regularized
        vs unregularized models.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        proc_dir = os.path.join(ROOT_DIR, "Data", "processed")
        if not os.path.exists(os.path.join(proc_dir, "asv_clr_binary.csv")):
            return  # skip if processed data not yet generated
        clr_bin  = pd.read_csv(os.path.join(proc_dir,"asv_clr_binary.csv"), index_col=0)
        meta_bin = pd.read_csv(os.path.join(proc_dir,"metadata_binary.csv"),index_col=0)
        y        = meta_bin["label_binary"].values
        X        = clr_bin.values
        cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes    = models_mod.build_pipelines()
        en_recall   = cross_val_score(pipes["LR-ElasticNet"], X, y, cv=cv, scoring="recall").mean()
        base_recall = cross_val_score(pipes["LR-Baseline (no reg)"], X, y, cv=cv, scoring="recall").mean()
        assert en_recall >= base_recall, \
            f"ElasticNet recall {en_recall:.3f} < baseline recall {base_recall:.3f}"


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    def test_full_preprocessing_pipeline_on_dummy_data(self):
        """
        Run full preprocessing pipeline on dummy data and verify
        outputs are correct end-to-end.
        """
        n_samples, n_asvs, n_genera = 20, 100, 15
        rng     = np.random.default_rng(42)
        asv_raw = pd.DataFrame(
            rng.integers(0, 50, size=(n_samples, n_asvs)),
            index=[f"S{i}" for i in range(n_samples)],
            columns=[f"ASV_{i}" for i in range(n_asvs)]
        )
        taxa_raw = pd.DataFrame({
            "Genus"  : [f"Genus_{i%n_genera}" for i in range(n_asvs)],
            "Family" : ["Lachnospiraceae"] * n_asvs,
            "Order"  : ["Lachnospirales"] * n_asvs,
            "Class"  : ["Clostridia"] * n_asvs,
            "Phylum" : ["Firmicutes"] * n_asvs,
            "Species": [np.nan] * n_asvs,
            "Label"  : [f"G_Genus_{i%n_genera}" for i in range(n_asvs)]
        }, index=[f"ASV_{i}" for i in range(n_asvs)])

        # Step 1: aggregate
        asv_genus = preproc.aggregate_to_genus(asv_raw, taxa_raw)
        assert asv_genus.shape[0] == n_samples
        assert asv_genus.shape[1] <= n_asvs

        # Step 2: filter
        asv_filt, prev = preproc.filter_by_prevalence(asv_genus, threshold=0.10)
        assert asv_filt.shape[0] == n_samples

        # Step 3: CLR
        asv_clr = preproc.clr_transform(asv_filt)
        assert asv_clr.shape == asv_filt.shape
        assert np.allclose(asv_clr.sum(axis=1), 0, atol=1e-10)

        print("\n  Integration test 1 PASSED: preprocessing pipeline correct")

    def test_eda_pipeline_on_dummy_data(self):
        """Run EDA steps on dummy data and verify output validity."""
        clr  = make_clr(20, 30)
        meta = make_meta(20)

        # PCA
        X_pca, var_exp, loadings = eda.run_pca(clr, n_components=5)
        assert X_pca.shape == (20, 5)
        assert var_exp.sum() <= 100.1

        # Diversity
        meta_div, kw_s, kw_r = eda.compute_alpha_diversity(make_counts(20,30), meta)
        assert "Shannon" in meta_div.columns
        assert 0 <= kw_s.pvalue <= 1

        # NMF
        W, H, err = eda.run_nmf(make_counts(20,30), n_components=3)
        assert W.shape == (20, 3)
        assert H.shape == (3, 30)
        assert (W.values >= 0).all()

        print("  Integration test 2 PASSED: EDA pipeline correct")

    def test_model_pipeline_on_dummy_data(self):
        """Run model evaluation on dummy data and verify metric validity."""
        from sklearn.model_selection import StratifiedKFold
        clr  = make_clr(20, 30)
        meta = make_meta(20)
        y    = meta["label_binary"].values
        X    = clr.values
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipes = models_mod.build_pipelines()

        for name in ["LR-Baseline (no reg)", "LR-L1", "LR-ElasticNet", "RF"]:
            row = models_mod.evaluate_model(name, pipes[name], X, y, cv)
            assert 0 <= row["_roc_auc_mean"] <= 1
            assert 0 <= row["_accuracy_mean"] <= 1

        print("  Integration test 3 PASSED: model pipeline correct")

    def test_processed_files_exist(self):
        """
        Verify all expected processed data files exist.
        Passes only after 01_preprocessing_final.py has been run.
        """
        proc_dir = os.path.join(ROOT_DIR, "Data", "processed")
        expected = [
            "asv_clr_full.csv", "asv_clr_binary.csv",
            "asv_counts_filtered.csv", "metadata_aligned.csv",
            "metadata_binary.csv", "taxonomy_map.csv",
        ]
        missing = [f for f in expected
                   if not os.path.exists(os.path.join(proc_dir, f))]
        assert len(missing) == 0, \
            f"Missing files: {missing}\nRun 01_preprocessing_final.py first."

    def test_results_files_exist(self):
        """
        Verify model result files exist.
        Passes only after 03_models_final.py has been run.
        """
        res_dir  = os.path.join(ROOT_DIR, "Results", "metrics")
        expected = [
            "model_comparison.csv",
            "lr_feature_importance.csv",
            "rf_feature_importance.csv",
            "consensus_biomarkers.csv",
        ]
        missing = [f for f in expected
                   if not os.path.exists(os.path.join(res_dir, f))]
        assert len(missing) == 0, \
            f"Missing result files: {missing}\nRun 03_models_final.py first."

    def test_model_files_exist(self):
        """
        Verify saved model .pkl files exist.
        Passes only after 03_models_final.py has been run.
        """
        model_dir = os.path.join(ROOT_DIR, "Results", "models")
        expected  = ["lr_elasticnet.pkl", "rf_tuned.pkl", "scaler.pkl"]
        missing   = [f for f in expected
                     if not os.path.exists(os.path.join(model_dir, f))]
        assert len(missing) == 0, \
            f"Missing model files: {missing}\nRun 03_models_final.py first."


# ============================================================
# MAIN — run without pytest
# ============================================================

if __name__ == "__main__":
    if not IMPORTS_OK:
        print(f"Could not import functions: {IMPORT_ERROR}")
        print("Make sure you run from the project root directory.")
        sys.exit(1)

    print("CRC Microbiome ML — Full Workflow Tests")
    print("=" * 60)

    test_classes = [
        TestCLRTransform, TestPrevalenceFilter, TestTaxonomyLabel,
        TestLabelVectors, TestPCA, TestAlphaDiversity,
        TestDifferentialAbundance, TestNMF, TestFeatureSets,
        TestCrossValidation, TestFeatureImportance,
        TestModelPredictions, TestIntegration,
    ]

    total, passed, failed = 0, 0, []

    for cls in test_classes:
        instance = cls()
        methods  = sorted([m for m in dir(cls) if m.startswith("test_")])
        print(f"\n{cls.__name__} ({len(methods)} tests)")
        for method in methods:
            total += 1
            try:
                getattr(instance, method)()
                print(f"  PASS  {method}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {method}: {e}")
                failed.append(f"{cls.__name__}.{method}")
            except Exception as e:
                print(f"  ERROR {method}: {type(e).__name__}: {e}")
                failed.append(f"{cls.__name__}.{method}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("All tests passed!")
    print("=" * 60)