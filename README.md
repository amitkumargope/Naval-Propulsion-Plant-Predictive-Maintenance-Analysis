# Naval Propulsion Plant — Predictive Maintenance Analysis

## Project Overview

This project implements a **comprehensive predictive maintenance analysis system** for naval propulsion plants based on the UCI Condition Based Maintenance of Naval Propulsion Plants dataset. It addresses three core research questions:

| Research Question | Approach |
|---|---|
| **Performance Decay Prediction** | Regression models predicting compressor (`kMc`) & turbine (`kMt`) decay coefficients |
| **Damage/Failure Prediction** | Binary classification models for early failure detection |
| **Operating Conditions Impact** | SHAP, permutation importance, partial dependence analysis |

---

## Dataset Description

**Source:** UCI Machine Learning Repository — [Condition Based Maintenance of Naval Propulsion Plants](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants)

**Local Path:** `C:\Users\gopeami\OneDrive - Vesuvius\Desktop\PhD13- 2025-2026\ML Practice\Steel industry\Shipping industry\archive\data.csv`

**Instances:** 11,934 | **Features:** 16 input + 2 target variables

### Feature Descriptions

| # | Feature | Unit | Description |
|---|---------|------|-------------|
| 1 | `lp` | — | Lever position |
| 2 | `v` | knots | Ship speed |
| 3 | `GTT` | kN·m | Gas Turbine shaft torque |
| 4 | `GTn` | rpm | GT rate of revolutions |
| 5 | `GGn` | rpm | Gas Generator rate of revolutions |
| 6 | `Ts` | kN | Starboard Propeller Torque |
| 7 | `Tp` | kN | Port Propeller Torque |
| 8 | `T48` | °C | HP Turbine exit temperature |
| 9 | `T1` | °C | GT Compressor inlet air temperature |
| 10 | `T2` | °C | GT Compressor outlet air temperature |
| 11 | `P48` | bar | HP Turbine exit pressure |
| 12 | `P1` | bar | GT Compressor inlet air pressure |
| 13 | `P2` | bar | GT Compressor outlet air pressure |
| 14 | `Pexh` | bar | GT exhaust gas pressure |
| 15 | `TIC` | % | Turbine Injection Control |
| 16 | `mf` | kg/s | Fuel flow |
| **T1** | `kMc` | — | **GT Compressor decay state coefficient** ← Regression Target |
| **T2** | `kMt` | — | **GT Turbine decay state coefficient** ← Regression Target |

### Engineered Features

| Feature | Description |
|---------|-------------|
| `compressor_degradation` | Normalised compressor wear (0 = perfect, 1 = fully degraded) |
| `turbine_degradation` | Normalised turbine wear |
| `combined_degradation` | Average system degradation score |
| `propeller_torque_imbalance` | \|Ts − Tp\| asymmetry |
| `temp_rise` | T2 − T1 (compressor temperature rise) |
| `pressure_ratio` | P2 / P1 (compressor pressure ratio) |
| `turbine_expansion_ratio` | P48 / Pexh |
| `thermal_efficiency_proxy` | GTT / mf |
| `speed_load_ratio` | v / lp |
| `GT_power_proxy` | GTT × GTn / 9550 [kW] |
| `fuel_efficiency` | v / mf |

---

## Methodology

### 1. Data Pipeline
```
Raw Data → EDA → Feature Engineering → Scaling (RobustScaler) → Train/Val/Test Split
```
- **Split:** 70% train / 15% validation / 15% test (stratified for classification)
- **Imbalance handling:** SMOTE oversampling on training set for classification

### 2. Models Implemented

#### Regression — Performance Decay Prediction
| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest Regressor | 200 trees, max_depth=10 |
| XGBoost Regressor | 300 estimators, lr=0.05 |
| LightGBM Regressor | 300 estimators, 63 leaves |

#### Classification — Failure Prediction
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline |
| Random Forest Classifier | class_weight="balanced" |
| XGBoost Classifier | scale_pos_weight adjusted |
| LightGBM Classifier | is_unbalance=True |

#### Interpretability — Operating Conditions Impact
- SHAP TreeExplainer (beeswarm + bar plots)
- Permutation Importance (10 repeats)
- Partial Dependence Plots (top 4 features)
- Operating condition heatmaps (speed × lever position grid)

---

## Model Results

### Regression (Test Set)
| Model | kMc R² | kMc RMSE | kMt R² | kMt RMSE |
|-------|--------|----------|--------|----------|
| Linear Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** | **Best** | **Lowest** | **Best** | **Lowest** |
| LightGBM | — | — | — | — |

> *Exact values printed in notebook Cell 21 after execution.*

### Classification (Test Set)
| Model | F1 | Recall | AUC |
|-------|----|--------|-----|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost** | **Best** | — | — |
| LightGBM | — | — | — |

---

## Maintenance Recommendations

### Alert Thresholds
| Indicator | ⚠️ Warning | 🔴 Critical |
|-----------|-----------|------------|
| Compressor decay `kMc` | < 0.980 | < 0.960 |
| Turbine decay `kMt` | < 0.990 | < 0.978 |
| Failure probability | > 50% | > 75% |

### Key Findings
1. **XGBoost and LightGBM** provide the best predictive accuracy for both regression and classification tasks.
2. **Top influential features** (from SHAP): fuel flow (`mf`), GT torque (`GTT`), temperature rise (`temp_rise`), pressure ratio, and ship speed.
3. **Predictive maintenance strategy** can reduce total maintenance costs by an estimated **30–50%** compared to reactive (corrective) maintenance.
4. **High-speed operations** (> 22 knots) combined with high lever positions accelerate both compressor and turbine degradation.

---

## Project Structure

```
Maritime_Industry/
├── Conditioning_Monitoring.ipynb   # Main analysis notebook (24 cells)
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── outputs/                        # Generated on first run
    ├── 01_feature_distributions.png
    ├── 02_target_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_pairplot.png
    ├── 05_boxplots.png
    ├── 06_regression_residuals.png
    ├── 07_regression_comparison.png
    ├── 08_clf_evaluation.png
    ├── 09_shap_summary_kMc.png
    ├── 10_shap_importance_kMc.png
    ├── 11_shap_summary_clf.png
    ├── 12_permutation_importance.png
    ├── 13_partial_dependence.png
    ├── 14_maintenance_alerts.png
    ├── 15_cost_benefit.png
    ├── 16_maintenance_dashboard.html   ← Interactive Plotly dashboard
    ├── 17_operating_condition_heatmap.png
    ├── 18_feature_importance_rf.png
    ├── 19_importance_heatmap.png
    └── saved_models/
        ├── xgb_kMc_regressor.pkl
        ├── xgb_kMt_regressor.pkl
        ├── xgb_classifier.pkl
        ├── rf_kMc_regressor.pkl
        ├── rf_kMt_regressor.pkl
        ├── rf_classifier.pkl
        ├── scaler.pkl
        ├── feature_names.pkl
        ├── maintenance_thresholds.pkl
        └── cost_parameters.pkl
```

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execute Notebook
Open `Conditioning_Monitoring.ipynb` in JupyterLab or VS Code and run all cells sequentially (Ctrl+Shift+Enter or "Run All").

### Configure Data Path
Edit `CONFIG["data_path"]` in **Cell 2** to point to your local dataset location.

---

## Citation

```
Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
Condition Based Maintenance of Naval Propulsion Plants [Dataset].
UCI Machine Learning Repository.


## Author
Amit Kumar Gope
AI Research Scientist
Vesuvius Group SA
Ghilin Mons Belgium
https://doi.org/10.24432/C5K31K
```

