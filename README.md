# CRC Microbiome ML Project

**Predicting Colorectal Cancer from Gut Microbiome Data using Machine Learning**

Hetvi Trivedi · Pascal Hayward · Diksha Patel · Sarah Quan · Louise Morr · Khaled Al Duwaikat  

## Overview

Colorectal cancer (CRC) is one of the leading causes of cancer-related deaths worldwide. Current detection relies on colonoscopy, which is invasive, expensive, and underutilized. This project builds machine learning models that classify CRC patients from healthy individuals using gut microbiome sequencing data from stool samples — a non-invasive alternative with potential clinical applications.

We apply preprocessing, exploratory analysis, and binary classification (CRC vs. Healthy) to a dataset of 59 patient stool samples with 6,693 microbial sequences. Our best model (ElasticNet Logistic Regression) achieves **AUC = 0.963** and **90% accuracy** using 5-fold cross-validation, and identifies 13 consensus microbial biomarkers validated against published CRC literature.


## Project Structure

```
crc-microbiome-ml/
├── Code/
│   ├── 01_preprocessing_final.py   ← Data cleaning, CLR transformation
│   ├── 02_eda_pca.py               ← PCA, diversity, differential abundance, NMF
│   └── 03_models_final.py          ← ML models, feature importance, confounding
│
├── Data/
│   ├── raw/                        ← Original data files (not tracked by Git)
│   │   ├── seqtab_nochim_export.xlsx
│   │   ├── taxa_species_export.xlsx
│   │   └── metadata.csv
│   └── processed/                  ← Outputs from preprocessing (not tracked)
│       ├── asv_clr_full.csv
│       ├── asv_clr_binary.csv
│       ├── asv_counts_filtered.csv
│       ├── asv_raw_asvlevel.csv
│       ├── metadata_aligned.csv
│       ├── metadata_binary.csv
│       ├── taxonomy_map.csv
│       ├── differential_abundance.csv
│       ├── nmf_sample_weights.csv
│       ├── nmf_genus_weights.csv
│       └── pca_loadings.csv
│
├── Results/
│   ├── figures/                    ← All generated plots (PNG)
│   ├── metrics/                    ← Model results and feature importance (CSV)
│   └── models/                     ← Saved model objects (.pkl)
│
├── Tests/
│   └── test_preprocessing.py       ← Unit tests for core functions
│
├── Notebooks/                      ← Exploratory prototyping (not final code)
│
├── .gitignore
└── README.md
```

---

## Dataset

The dataset is sourced from [Kaggle — CRC Gut Microbiome ML Data](https://www.kaggle.com/datasets/aramelheni/crc-gut-microbiome-ml-data).

| File | Description |
|---|---|
| `seqtab_nochim_export.xlsx` | 59 samples × 6,693 ASV count matrix |
| `taxa_species_export.xlsx` | Taxonomy table (Kingdom → Species) for each ASV |
| `metadata.csv` | Patient metadata: DiseaseStatus, Age, Sex |

**Groups:** 21 Colorectal Cancer · 19 Healthy · 19 Adenomatous Polyps  
**ML task:** Binary classification — CRC vs. Healthy (40 samples)

> ⚠️ Raw data files are not tracked by Git due to file size. Place them in `Data/raw/` before running any scripts.

---

## Dependencies

Python 3.9+ is required. All dependencies can be installed via pip.

**Recommended: use a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**Install packages**

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn openpyxl joblib
```

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥1.5 | Data loading, manipulation, CSV I/O |
| numpy | ≥1.23 | Matrix operations, CLR transformation |
| scikit-learn | ≥1.1 | PCA, NMF, models, cross-validation |
| scipy | ≥1.9 | Statistical tests (Mann-Whitney, Kruskal-Wallis, FDR) |
| matplotlib | ≥3.6 | Figures and plots |
| seaborn | ≥0.12 | Heatmaps and styled plots |
| openpyxl | ≥3.0 | Reading Excel (.xlsx) files |
| joblib | ≥1.2 | Saving and loading model objects |

---

## How to Run

Scripts must be run **in order** — each script depends on outputs from the previous one.

### Step 1 — Preprocessing

```bash
python3 Code/01_preprocessing_final.py
```

**What it does:**
- Loads raw ASV table, taxonomy, and metadata
- Aggregates 6,693 ASVs → 277 genera → 122 genera (after prevalence filtering)
- Applies CLR transformation to correct for compositional bias
- Saves processed CSVs to `Data/processed/`
- Generates QC figures in `Results/figures/`

**Expected runtime:** 2–4 minutes (Excel file loading is slow)

---

### Step 2 — Exploratory Analysis

```bash
python3 Code/02_eda_pca.py
```

**What it does:**
- PCA visualization (all 3 groups including Adenomatous Polyps)
- Alpha diversity analysis (Shannon Index, Richness)
- Differential abundance testing (Mann-Whitney U, BH FDR correction)
- NMF microbial community discovery (5 components)
- Literature validation of key significant genera
- Saves figures and summary metrics

**Expected runtime:** Under 1 minute

---

### Step 3 — Machine Learning Models

```bash
python3 Code/03_models_final.py
```

**What it does:**
- Evaluates 5 model types: unregularized baseline, LR-L1, LR-L2, LR-ElasticNet, Random Forest
- Stratified 5-fold cross-validation throughout
- ROC curve comparison (ElasticNet vs baseline)
- Feature importance extraction and consensus biomarker identification
- Confounding check (age and sex as predictors)
- Saves all results, figures, and model objects

**Expected runtime:** 2–5 minutes (Random Forest with 500 trees)

---

### Step 4 — Run Unit Tests

```bash
python3 -m pytest Tests/ -v
```

---

## Key Results

| Model | ROC-AUC | Accuracy | F1 | Recall |
|---|---|---|---|---|
| **LR-ElasticNet** | **0.963** | **90.0%** | **0.923** | **100%** |
| LR-L1 (tuned) | 0.950 | 87.5% | 0.877 | 86% |
| LR-Baseline (no reg) | 0.963 | 82.5% | 0.845 | 87% |
| RF (top20 DA) | 0.925 | 87.5% | 0.895 | 95% |

**Key biological finding:** Butyrate-producing bacteria (*Fusicatenibacter*, *Anaerostipes*, *Lachnospira*) are consistently depleted in CRC patients — consistent with Wirbel et al. 2019 (Nature Medicine) and Zeller et al. 2014 (Molecular Systems Biology).

**Confounding:** Demographics-only model (age + sex) achieves AUC = 0.219, confirming the microbiome signal is not driven by demographic confounders.

---

## Methods Summary

### Preprocessing
- ASV → genus aggregation via taxonomy table (unbiased, rule-based)
- Prevalence filtering at 10% threshold (standard in microbiome literature)
- CLR transformation to correct compositional bias (Gloor et al. 2017)

### Exploratory Analysis
- PCA with StandardScaler on CLR data (all 3 groups)
- Alpha diversity: Shannon Index and Richness (Kruskal-Wallis test)
- Differential abundance: Mann-Whitney U + Benjamini-Hochberg FDR correction
- NMF (5 components) for microbial community discovery

### Machine Learning
- Binary task: CRC vs. Healthy (40 samples)
- Adenomatous Polyps excluded: only 19 samples per class — too small for reliable 3-class ML
- Stratified 5-fold cross-validation (n=40 too small for held-out test set)
- Logistic Regression variants: no regularization (baseline), L1, L2, ElasticNet
- Random Forest: 500 trees, class_weight="balanced"

---

## Team Contributions

| Member | Role |
|---|---|
| Hetvi Trivedi | EDA script, PCA analysis, models script, presentation, README |
| Pascal Hayward | Preprocessing pipeline, CLR implementation |
| Diksha Patel | Feature engineering, NMF analysis |
| Sarah Quan | Model evaluation, cross-validation framework |
| Louise Morr | Visualization, figures |
| Khaled Al Duwaikat | Biological interpretation, report writing |

---

## Literature References

- Wirbel J. et al. (2019). Meta-analysis of fecal metagenomes reveals global microbial signatures that are specific for colorectal cancer. *Nature Medicine*, 25, 679–689.
- Zeller G. et al. (2014). Potential of fecal microbiota for early-stage detection of colorectal cancer. *Molecular Systems Biology*, 10, 766.
- Gloor G.B. et al. (2017). Microbiome datasets are compositional: and this is not optional. *Frontiers in Microbiology*, 8, 2224.

---

## Notes

- Data files are excluded from Git tracking (see `.gitignore`) due to file size limits
- All scripts include a `BASE` path variable at the top — update this to match your local directory before running
- Virtual environment (`venv/`) is also excluded from Git tracking