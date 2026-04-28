# Predictive Analytics for Student Success (OUALD)
[![Python](https://img.shields.io/badge/Language-Python-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg?style=flat&logo=pandas)](https://pandas.pydata.org/)

**Copyright © Springer Robert Stefan, 2026**

> An end-to-end Machine Learning ecosystem designed to predict academic outcomes using the Open University Learning Analytics Dataset (OUALD).
> 
> This system implements a high-performance pipeline: from advanced data orchestration and hybrid imputation to multi-model evaluation achieving good results by balancing architectural bias and variance.

## **Table of Contents**
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Orchestration Pipeline](#data-orchestration-pipeline)
- [Analysis & Modeling](#analysis--modeling)
- [Performance Results](#performance-results)

## **Overview**

This project analyzes student interactions within Virtual Learning Environments (VLE) to provide high-fidelity early predictions:
1. **Classification**: Predicts categorical outcomes (`Pass`, `Fail`, `Withdrawn`, `Distinction`).
2. **Regression**: Estimates precise performance scores (`final_coursework_score`, range 0-100).

## **System Architecture**

| Module | Purpose |
|:---:|:---|
| **`main.py`** | Central orchestrator managing the full lifecycle from data ingestion to model deployment. |
| **`DataAnalysis.py`** | Exploratory Data Analysis (EDA) engine using Pearson correlation and Chi-Square statistical tests. |
| **`DataProcessing.py`** | Advanced cleaning, IQR-based outlier management (0.05-0.95), and automated feature engineering. |
| **`Clasify.py`** | Classification logic featuring Hyperparameter-tuned Random Forest and Decision Tree architectures. |
| **`Regression.py`** | Comparative regression engine evaluating Linear, Ridge, and Ensemble methods. |

## **Data Orchestration Pipeline**

The pipeline utilizes production-grade methods to elevate baseline accuracy from ~50% to over 76%:

1. **Outlier Mitigation**: Implements the Interquartile Range (IQR) method with optimized thresholds (q1=0.05, q3=0.95) to preserve meaningful extremes while removing noise.
2. **Hybrid Imputation Strategy**: 
   * **Numeric Features**: Statistical `Mean` imputation to maintain data distribution.
   * **Categorical Features**: `Most Frequent` (Mode) strategy to preserve class integrity.
3. **Feature Engineering & Dimensionality Reduction**:
   * **Total Activity Synthesis**: Aggregates disparate VLE interaction metrics into a unified `total_activity` score.
   * **Categorical Mapping**: Efficiently maps ordinal features (Age, IMD, Education) to numeric scales for optimized computation.
4. **Redundancy Filtering**: Automated drops for attributes with cross-correlation > 0.9 (e.g., *submission_rate_early*).

## **Analysis & Modeling**

### **Classification (Ensemble Methods)**
- **Hyperparameters**: `n_estimators=250`, `min_samples_leaf=4`.
- **Key Metrics**: Accuracy: **0.7649** | Weighted F1-Score: **0.7482**.
- **Insight**: The model exhibits exceptional performance in identifying "Withdrawn" status through the hybrid prediction logic.

### **Regression (Predictive Scoring)**
- **Hyperparameters**: `n_estimators=100`, `max_depth=8`.
- **Key Metrics**: R² Score: **0.8092** | MAE: **4.9736**.
- **Insight**: Learning Curve analysis demonstrates that while linear models suffer from underfitting, the Random Forest architecture scales effectively with data volume for superior generalization.

## **Performance Results**

| Model | Accuracy / R² | Precision / MAE |
|:---|:---:|:---:|
| **RandomForest Classifier** | **76.49%** | 0.7702 |
| **DecisionTree Classifier** | 74.53% | 0.7430 |
| **RandomForest Regressor** | **0.8092 (R²)** | 4.9736 (MAE) |
| **Linear Regression** | 0.8054 (R²) | 5.0943 (MAE) |

## **Setup & Usage**

### **Installation**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### **Execution**
```bash
python3 main.py
```
