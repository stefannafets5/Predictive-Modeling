# **Student Success Prediction System**
[![Python](https://img.shields.io/badge/Language-Python-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg?style=flat&logo=pandas)](https://pandas.pydata.org/)

**Copyright © Springer Robert Stefan, 2026**

> An end-to-end **Machine Learning Pipeline** designed to predict student performance using the Open University Learning Analytics Dataset (OUALD).
> 
> The system implements a robust data processing engine and evaluates multiple algorithms for both **Classification** (predicting final results) and **Regression** (predicting coursework scores), achieving an optimized balance between bias and variance.

## **Table of Contents**
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Analysis & Modeling](#analysis--modeling)
- [Key Features](#key-features)
- [Setup & Usage](#setup--usage)

## **Overview**

The project aims to analyze student interaction data with a Virtual Learning Environment (VLE) to provide early predictions on:
1.  **Classification**: `final_result` (Pass, Fail, Withdrawn, Distinction).
2.  **Regression**: `final_coursework_score` (Target score from 0-100).

The dataset is divided into training, validation, and test sets, ensuring that models are evaluated on unseen data to test generalization capabilities.

## **Project Architecture**

| Module | Description |
|:---:|:---|
| **`main.py`** | The central orchestrator for data loading, analysis, processing, and model execution. |
| **`DataAnalysis.py`** | Performs Exploratory Data Analysis (EDA), including boxplots, countplots, and correlation matrices. |
| **`DataProcessing.py`** | Handles cleaning, outlier removal (IQR), hybrid imputation, and feature engineering. |
| **`Clasify.py`** | Implements classification logic using RandomForest and DecisionTree with hyperparameter tuning. |
| **`Regression.py`** | Compares LinearRegression, Ridge, and Random Forest for score prediction. |

## **Data Processing Pipeline**

To achieve high accuracy (~62% classification and 0.75 R² regression), the following steps are performed:

1.  **Outlier Cleaning**: Uses the Interquartile Range (IQR) method to replace extreme values with `NaN`.
2.  **Hybrid Imputation**:
    *   **Numeric**: Replaces `NaN` with the `Mean` to maintain distribution.
    *   **Categorical**: Replaces `NaN` with the `Mode` (Most Frequent).
3.  **Feature Engineering**:
    *   Merges various interaction types into a single `total_activity` metric.
    *   Maps ordinal categorical features (Age, IMD, Education) to numeric scales.
4.  **Standardization**: Applies `StandardScaler` to numeric attributes to ensure models aren't biased by feature magnitude.

## **Analysis & Modeling**

### **Classification Logic**
- **Balanced Weights**: Uses `class_weight='balanced'` in Random Forest to handle the inherent imbalance in student results (e.g., few Distinction cases).
- **Evaluation**: Tracks Accuracy, Weighted Precision, Recall, and F1-Score.
- **Confusion Matrix**: Visualizes exactly where the model confuses "Pass" with "Distinction" or "Fail".

### **Regression Logic**
- **Baseline vs. Advanced**: Compares simple `LinearRegression` with `RandomForestRegressor`.
- **Metrics**: Evaluates performance using MAE, MSE, RMSE, and R².
- **Learning Curves**: Tracks error evolution relative to training set size to detect overfitting.

## **Key Features**

*   **Robust Preprocessing**: Automatic identification and removal of redundant attributes with correlation scores > 0.8.
*   **Chi-Square Validation**: Statistically assesses the relationship between categorical features and the target class.
*   **Visual Documentation**: Automatically generates PNG reports (scatter plots, confusion matrices, and learning curves) for every execution.
*   **Performance Tracking**: Maintains a record of the "Best Average Score" across different hyperparameter configurations (leaf size, depth).

## **Setup & Usage**

### **Dependencies**
Install the required libraries using pip then run main.py:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy

python3 main.py