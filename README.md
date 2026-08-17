# Supervised ML Benchmark — Telco Customer Churn

## Overview
This project benchmarks multiple supervised classification algorithms on the public Telco Customer Churn dataset.

## Models
- Logistic Regression
- K-Nearest Neighbors (K=3 and K=7)
- Decision Tree (shallow and deep)
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (with and without feature scaling)

## Preprocessing
- `TotalCharges` is converted to numeric; non-convertible values become missing values.
- `customerID` is removed because it is an identifier rather than a predictive feature.
- Numerical features use median imputation and StandardScaler.
- Categorical features use most-frequent imputation and OneHotEncoder.
- Preprocessing is placed inside sklearn pipelines to prevent data leakage.

## Evaluation
Models are compared using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- Training time
- Inference time

The notebook uses a fixed random seed and stratified train/validation/test splits.

## How to run

Expected project structure:

```text
assignment_1/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── supervised_ml_benchmark_completed.ipynb
├── README.md
└── observations.md
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Run:

```bash
jupyter notebook
```

Open `supervised_ml_benchmark_completed.ipynb` and run all cells from top to bottom.

## Important
The final test set is evaluated only after validation-based model comparison. The notebook generates the validation and test result tables automatically so metrics are not manually entered.
