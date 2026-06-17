# Lead Scoring ML System — 86% AUC-ROC, Identifying Buyers Before They Raise Their Hand

> **86% AUC-ROC** on held-out validation data · **5 identified customer segments** · End-to-end ML pipeline from raw CRM data to pre-production deployment

---

## Why This Project Exists

Most CRM systems generate leads. Few can tell you *which* leads are worth your sales team's time. This project builds an ML system that predicts the probability of a lead converting into a customer — and then segments those leads into actionable behavioral profiles to prioritize outreach.

The same logic applies far beyond sales. Predicting which companies will adopt a sustainability reporting standard, which suppliers carry ESG risk, or which assets are misaligned with SFDR disclosure requirements is structurally the same problem: **binary classification over behavioral signals**. This project is my foundation for that work.

---

## Dataset

- **Source:** CRM export (`Leads.csv`)
- **Size:** 9,093 leads · 20 features
- **Target:** `compra` (purchase/conversion) — 37.6% positive rate (no severe class imbalance)
- **Feature types:** behavioral (time on site, page views, activity score), demographic (occupation, sector), and channel (source, last activity, lead magnet download)

---

## What Was Built — End-to-End ML Pipeline

### 1 · Data Import & Leakage-Free Split
Raw CRM data ingested from CSV. A **30% holdout validation set** is separated before any transformation — the same split is used for final model evaluation, ensuring no data leakage.

### 2 · Data Quality
Systematic quality audit across all 20 variables:
- Type corrections (e.g., `visitas_total` → `Int64`)
- Zero-variance column removal (`conociste_youtube`, `conociste_revista`)
- **Duplicate removal**
- **Business logic filters:** leads who opted out of email contact or whose emails bounced are excluded — they cannot be reached, so they must not inflate model performance
- Missing value strategy: mode imputation for categoricals (`ocupacion`, `ambito`), median imputation for numerics, `"DESCONOCIDO"` category for unknowns — designed to be **production-safe** (won't break on new nulls)
- **Winsorization** on `visitas_total` (cap: 50) and `paginas_vistas_visita` (cap: 20)
- Rare category grouping (< 2% frequency → `"OTROS"`)

### 3 · Exploratory Data Analysis

Key findings that shaped the model:
- `tiempo_en_site_total` is **bimodal** — two distinct engagement patterns exist in the data
- `score_actividad` and `score_perfil` appear pre-calculated by the CRM, their outliers are meaningful and are preserved
- **No severe imbalance** (37.6% / 62.4%) — standard metrics apply without oversampling

### 4 · Feature Engineering & Transformation
All categorical variables encoded via **One-Hot Encoding** (no ordinal assumptions imposed). Numerical variables scaled with **Min-Max Scaling** [0–1] — chosen deliberately over Standard Scaling because clustering requires all features in the same range.

### 5 · Customer Segmentation (K-Means Clustering — Unsupervised)
Before the predictive model, leads are segmented to understand behavioral archetypes. Four metrics evaluated simultaneously: **Elbow, Silhouette, Calinski-Harabasz, and Davies-Bouldin**.

**Optimal solution: K = 5 segments**, with the most striking finding being **segment 3 achieving ~90% conversion rate** — a high-intent cluster identifiable before any sales contact.

Segment profiles defined along dimensions of:
- Traffic source (chat/API vs. organic vs. referral)
- Occupation (working professional vs. unemployed)
- Lead magnet download behavior
- Last activity type

### 6 · Feature Selection
Three independent methods applied and cross-validated:
- **Mutual Information** — `tiempo_en_site_total`, `score_actividad`, `ult_actividad_SMS_Sent`, `ocupacion_Working_Professional` emerge as top signals
- **Recursive Feature Elimination (RFE)** with XGBoost as estimator
- **Permutation Importance** — confirms the same variable ranking, compresses the signal into fewer features

All three methods converge on the same top predictors — a strong signal that the feature selection is robust. **13 final features** selected; correlated variables removed to avoid multicollinearity.

### 7 · Modeling — Logistic Regression with GridSearchCV

Algorithm: **Logistic Regression** (interpretable, production-deployable, coefficient-reportable)

Hyperparameter optimization via **3-fold cross-validated GridSearchCV** over:
- Penalty type: `elasticnet`, `l1`, `l2`, `none`
- Regularization strength C: `[0, 0.25, 0.5, 0.75, 1]`
- Solver: `saga`

**Result: AUC-ROC = 0.86 on held-out validation set.**

Top predictive features (by coefficient magnitude):
1. `tiempo_en_site_total_mms` — time spent on site is the strongest conversion signal
2. `score_actividad_mms` — CRM activity score
3. `paginas_vistas_visita_mms` — pages per visit (negative coefficient: concentrated sessions convert better than scattered browsing)

Model reporting includes:
- **Cumulative Gain Chart** — shows the model captures ~80% of buyers by contacting the top 40% of leads
- **Lift Chart** — quantifies how much better the model is than random selection at each decile
- **ROC Curve** — plotted independently of `scikit-plot` for NumPy 2.0 compatibility

### 8 · Pre-Production Pipeline (In Progress)

Scikit-learn `Pipeline` + `ColumnTransformer` architecture being finalized to consolidate all transformation steps into a single, serializable object:
- Input: raw CRM record
- Output: conversion probability score
- Components: quality functions → OrdinalEncoder → MinMaxScaler → LogisticRegression
- Remaining work: deployment script, retraining script, input validation layer

---

## Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy`, `pyjanitor` |
| ML & preprocessing | `scikit-learn`, `xgboost` |
| Visualization | `matplotlib`, `seaborn`, `scikitplot` |
| Serialization | `pickle` |
| Environment | `conda`, Python 3.11 |

---

## Project Structure

```
├── 02_datos/
│   ├── 01_Originales/          # Raw CRM data
│   ├── 02_Validacion/          # Held-out validation set (created before any processing)
│   └── 03_Entrenamiento/       # Processed intermediate datasets (.pkl checkpoints)
└── 03_notebooks/
    ├── 01_Importacion_Datos    # Data ingestion & train/validation split
    ├── 02_Calidad_Datos        # Data quality audit & cleaning
    ├── 03_EDA                  # Exploratory data analysis
    ├── 04_Transformacion_Datos # Encoding & scaling
    ├── 05_Clustering           # K-Means segmentation
    ├── 06_Seleccion_Variables  # Feature selection (MI + RFE + Permutation)
    ├── 07_Modelizacion         # Model training, tuning & reporting
    └── 08_Preproduccion        # Production pipeline (in progress)
```

---

## What This Transfers To

This is not a sales project. It is a classification problem over behavioral signals — a pattern that recurs in ESG and sustainability contexts:

- **ESG Provider Scoring:** classify ESG data providers by signal quality using behavioral and disclosure patterns
- **CSRD Readiness Assessment:** predict which companies in a portfolio are likely to be compliant or non-compliant
- **Supplier Risk Classification:** score suppliers by sustainability risk using activity and reporting signals
- **Green Loan Eligibility:** rank applicants by alignment with EU Taxonomy criteria

The pipeline architecture — quality → EDA → transformation → segmentation → feature selection → classification → production — is the same regardless of domain. What changes is the data and the business question.

---

## Status

`IN PROGRESS` — Core ML pipeline complete and validated. Deployment, retraining, and production scripts pending.

---

## Author

**David Santos** · ESG Data Scientist  
[github.com/Daaviid99](https://github.com/Daaviid99) · [davidsantossalvador.es/blog](https://davidsantossalvador.es/blog)