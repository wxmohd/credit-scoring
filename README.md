# Credit Scoring Project

## Overview

This project implements a credit scoring model to predict the probability of
loan default for Home Credit customers.  The model uses LightGBM with feature
engineering across 7 data sources and provides interpretable predictions via
SHAP values.

**Kaggle username**: `username01EDU location_03_2026`

## Project Structure

```
credit-scoring/
├── data/                                   # Raw CSV datasets
│   └── DATA_INFO.md
├── results/
│   ├── model/
│   │   ├── my_own_model.pkl               # Trained LightGBM model
│   │   └── model_report.txt
│   ├── feature_engineering/
│   │   ├── EDA.ipynb
│   │   ├── feature_importance.png
│   │   └── learning_curves.png
│   └── clients_outputs/
│       ├── client1_correct_train.pdf
│       ├── client2_wrong_train.pdf
│       └── client_test.pdf
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── README.md
├── requirements.txt
└── username.txt
```

## Dataset

Home Credit Default Risk – 8 CSV files linked via `SK_ID_CURR`:

| File | Description |
|------|-------------|
| `application_train/test.csv` | Main table – TARGET in train only |
| `bureau.csv` | Previous credits from other institutions |
| `bureau_balance.csv` | Monthly balance history of bureau credits |
| `previous_application.csv` | Previous Home Credit loan applications |
| `POS_CASH_balance.csv` | Monthly POS / cash loan balances |
| `credit_card_balance.csv` | Monthly credit card balances |
| `installments_payments.csv` | Repayment history |
| `HomeCredit_columns_description.csv` | Column descriptions |

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Exploratory Data Analysis

```bash
jupyter notebook results/feature_engineering/EDA.ipynb
```

### 2. Train the Model

```bash
python scripts/train.py
```

Outputs saved to `results/model/` and `results/feature_engineering/`.

### 3. Generate Predictions & Client Reports

```bash
python scripts/predict.py
```

Expected output:
```
AUC on test set: 0.XX
```

Outputs saved to `results/clients_outputs/` and `results/submission.csv`.

## Model

- **Algorithm**: LightGBM (gradient boosting)
- **Validation**: 5-fold Stratified K-Fold cross-validation
- **Metric**: AUC-ROC (accuracy not used – class imbalance ~8 % default)
- **Target AUC**: ≥ 62 %
- **Overfitting prevention**: early stopping (100 rounds), L1/L2 regularization,
  row/feature subsampling

See `results/model/model_report.txt` for full methodology.

## Interpretability

- **Global**: Feature importance plot (top 30 features by split count)
- **Local**: SHAP waterfall plots per client

## Kaggle Username

`username01EDU location_03_2026`
