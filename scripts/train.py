import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, build_features

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, 'results', 'model')
FEATURE_DIR = os.path.join(BASE_DIR, 'results', 'feature_engineering')
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',
    'n_estimators':     5000,
    'learning_rate':    0.02,
    'num_leaves':       34,
    'max_depth':        -1,
    'min_child_samples': 20,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'min_split_gain':   0.01,
    'verbose':          -1,
    'random_state':     42,
    'n_jobs':           -1,
}


def plot_feature_importance(model, feature_names, top_n=30):
    """Plot and save top-N feature importance."""
    imp = pd.DataFrame({'feature': feature_names,
                        'importance': model.feature_importances_}
                       ).sort_values('importance', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(imp['feature'][::-1], imp['importance'][::-1], color='steelblue')
    ax.set_xlabel('Importance (split count)')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    path = os.path.join(FEATURE_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved to {path}")
    return imp


def plot_learning_curves(train_aucs, val_aucs):
    """Plot training vs validation AUC curves."""
    rounds = range(1, len(train_aucs) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_aucs,  label='Train AUC',      color='blue',   alpha=0.7)
    ax.plot(rounds, val_aucs,    label='Validation AUC', color='orange', alpha=0.7)
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('AUC')
    ax.set_title('Learning Curves (Training vs Validation AUC)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FEATURE_DIR, 'learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {path}")


def train_model(X, y, X_test):
    """5-fold CV LightGBM with early stopping."""
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    best_models, fold_scores = [], []
    first_train_aucs, first_val_aucs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/5 ---")
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ]
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  eval_metric='auc',
                  callbacks=callbacks)

        if fold == 1:
            evals = model.evals_result_
            first_train_aucs = evals['training']['auc']
            first_val_aucs   = evals['valid_1']['auc']

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds        += model.predict_proba(X_test)[:, 1] / 5
        best_models.append(model)

        score = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(score)
        print(f"Fold {fold} AUC: {score:.4f}  (best iter: {model.best_iteration_})")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.4f}")
    print(f"Mean Fold AUC: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    best_idx   = int(np.argmax(fold_scores))
    best_model = best_models[best_idx]
    print(f"Best fold: {best_idx + 1} (AUC={fold_scores[best_idx]:.4f})")

    plot_learning_curves(first_train_aucs, first_val_aucs)
    plot_feature_importance(best_model, X.columns.tolist())
    return best_model, oof_preds, test_preds, oof_auc


RAW_COLS = [
    'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CODE_GENDER',
    'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'REGION_RATING_CLIENT',
]


if __name__ == "__main__":
    train_raw, test_raw, bureau, bureau_balance, prev, pos, cc, inst = load_data()
    train_df, test_df = build_features(
        train_raw, test_raw, bureau, bureau_balance, prev, pos, cc, inst
    )

    y         = train_df['TARGET']
    ids_train = train_df['SK_ID_CURR']
    ids_test  = test_df['SK_ID_CURR']
    X         = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    X_test    = test_df.drop(columns=['SK_ID_CURR'])

    common_cols = [c for c in X.columns if c in X_test.columns]
    X      = X[common_cols]
    X_test = X_test[common_cols]
    print(f"\nFeature matrix: {X.shape}")

    best_model, oof_preds, test_preds, oof_auc = train_model(X, y, X_test)

    with open(os.path.join(MODEL_DIR, 'my_own_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved.")

    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(common_cols, f)

    val_data = {'oof_preds': oof_preds, 'y_true': y.values,
                'ids': ids_train.values, 'oof_auc': oof_auc}
    with open(os.path.join(MODEL_DIR, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    sub = pd.DataFrame({'SK_ID_CURR': ids_test, 'TARGET': test_preds})
    sub.to_csv(os.path.join(BASE_DIR, 'results', 'submission.csv'), index=False)
    print("Submission saved.")

    train_raw_cols = RAW_COLS + ['TARGET']
    test_raw_cols  = RAW_COLS
    sample_data = {
        'X': X, 'y': y.values, 'ids': ids_train.values,
        'X_test': X_test, 'ids_test': ids_test.values,
        'train_raw': train_raw[[c for c in train_raw_cols if c in train_raw.columns]].copy(),
        'test_raw':  test_raw[[c  for c in test_raw_cols  if c in test_raw.columns]].copy(),
    }
    with open(os.path.join(MODEL_DIR, 'processed_data.pkl'), 'wb') as f:
        pickle.dump(sample_data, f)
    print("Processed data saved.")
    print(f"\n=== Training complete – OOF AUC: {oof_auc:.4f} ===")
