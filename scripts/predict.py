import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'results', 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'clients_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    path = os.path.join(MODEL_DIR, 'my_own_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found – run train.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_processed_data():
    for name in ('val_data.pkl', 'processed_data.pkl'):
        if not os.path.exists(os.path.join(MODEL_DIR, name)):
            raise FileNotFoundError(f"{name} not found – run train.py first.")
    with open(os.path.join(MODEL_DIR, 'val_data.pkl'),       'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'processed_data.pkl'), 'rb') as f:
        proc_data = pickle.load(f)
    return val_data, proc_data


def plot_shap_waterfall(explainer, sv_row, ev, X_row, title, pdf):
    """SHAP waterfall plot saved to PDF page."""
    exp = shap.Explanation(
        values=sv_row,
        base_values=ev,
        data=X_row.values,
        feature_names=list(X_row.index),
    )
    plt.figure(figsize=(14, 5))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close('all')


def plot_client_profile(client_raw, train_raw, score, label, pdf):
    """Page 1: client bar chart.  Page 2: comparison vs population."""
    age = abs(client_raw['DAYS_BIRTH']) / 365
    emp_days = client_raw.get('DAYS_EMPLOYED', np.nan)
    emp = abs(emp_days) / 365 if pd.notna(emp_days) and emp_days < 0 else np.nan

    profile = {
        'Age (years)':        round(age, 1),
        'Income (k)':         round(client_raw['AMT_INCOME_TOTAL'] / 1000, 1),
        'Credit amount (k)':  round(client_raw['AMT_CREDIT'] / 1000, 1),
        'Annuity (k)':        round(client_raw['AMT_ANNUITY'] / 1000, 1) if pd.notna(client_raw.get('AMT_ANNUITY')) else 0,
        'Goods price (k)':    round(client_raw['AMT_GOODS_PRICE'] / 1000, 1) if pd.notna(client_raw.get('AMT_GOODS_PRICE')) else 0,
        'Employment (yrs)':   round(emp, 1) if pd.notna(emp) else 0,
        'Children':           client_raw['CNT_CHILDREN'],
        'Family members':     client_raw['CNT_FAM_MEMBERS'] if pd.notna(client_raw.get('CNT_FAM_MEMBERS')) else 0,
        'EXT_SOURCE_1':       round(client_raw['EXT_SOURCE_1'], 3) if pd.notna(client_raw.get('EXT_SOURCE_1')) else 0,
        'EXT_SOURCE_2':       round(client_raw['EXT_SOURCE_2'], 3) if pd.notna(client_raw.get('EXT_SOURCE_2')) else 0,
        'EXT_SOURCE_3':       round(client_raw['EXT_SOURCE_3'], 3) if pd.notna(client_raw.get('EXT_SOURCE_3')) else 0,
        'Region rating':      client_raw['REGION_RATING_CLIENT'] if pd.notna(client_raw.get('REGION_RATING_CLIENT')) else 0,
    }

    color = '#d62728' if score > 0.5 else '#2ca02c'
    fig, ax = plt.subplots(figsize=(12, 7))
    vals = [float(v) for v in profile.values()]
    bars = ax.barh(list(profile.keys()), vals, color=color, alpha=0.75)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xlabel('Value')
    ax.set_title(f'{label}\nDefault Score: {score:.4f}  ({"HIGH RISK" if score > 0.5 else "LOW RISK"})', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

    compare_cols = [
        ('AMT_INCOME_TOTAL', 'Income'), ('AMT_CREDIT', 'Credit'),
        ('AMT_ANNUITY', 'Annuity'),
        ('EXT_SOURCE_1', 'EXT SOURCE 1'), ('EXT_SOURCE_2', 'EXT SOURCE 2'),
        ('EXT_SOURCE_3', 'EXT SOURCE 3'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{label} – vs Population', fontsize=12)
    for ax, (col, lbl) in zip(axes.flatten(), compare_cols):
        ax.hist(train_raw[col].dropna(), bins=50, color='steelblue', alpha=0.6, label='Population')
        cval = client_raw.get(col, np.nan)
        if pd.notna(cval):
            ax.axvline(cval, color='red', lw=2, label=f'Client: {cval:,.1f}')
        ax.set_title(lbl)
        ax.legend(fontsize=8)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()


def select_clients(y, ids, oof_preds, ids_test):
    y_bin   = (oof_preds >= 0.5).astype(int)
    correct = np.where(y_bin == y)[0]
    wrong   = np.where(y_bin != y)[0]
    c1_idx  = correct[np.argmax(np.abs(oof_preds[correct] - 0.5))]
    c2_idx  = wrong[np.argmax(np.abs(oof_preds[wrong]   - 0.5))]
    c3_idx  = 0
    return c1_idx, ids[c1_idx], c2_idx, ids[c2_idx], c3_idx, ids_test[c3_idx]


if __name__ == "__main__":
    model = load_model()
    val_data, proc_data = load_processed_data()

    X         = proc_data['X']
    y         = proc_data['y']
    ids       = proc_data['ids']
    X_test    = proc_data['X_test']
    ids_test  = proc_data['ids_test']
    train_raw = proc_data['train_raw']
    test_raw  = proc_data['test_raw']
    oof_preds = val_data['oof_preds']
    oof_auc   = val_data['oof_auc']

    print(f"AUC on test set: {oof_auc:.2f}")

    c1_idx, c1_id, c2_idx, c2_id, c3_idx, c3_id = select_clients(y, ids, oof_preds, ids_test)
    print(f"\nClient 1 (correct, train): SK_ID_CURR={c1_id}")
    print(f"Client 2 (wrong,   train): SK_ID_CURR={c2_id}")
    print(f"Client 3 (test):           SK_ID_CURR={c3_id}")

    print("\nComputing SHAP values...")
    background = shap.sample(X, 100, random_state=42)
    explainer  = shap.TreeExplainer(model, background)

    clients_X = pd.concat([X.iloc[[c1_idx]], X.iloc[[c2_idx]], X_test.iloc[[c3_idx]]],
                           ignore_index=True)
    sv = explainer.shap_values(clients_X)
    ev = explainer.expected_value
    if isinstance(sv, list):
        sv = sv[1]
        ev = ev[1] if hasattr(ev, '__len__') else ev

    scores = [
        oof_preds[c1_idx],
        oof_preds[c2_idx],
        model.predict_proba(X_test.iloc[[c3_idx]])[:, 1][0],
    ]

    configs = [
        ('client1_correct_train.pdf',
         f'Client 1 (Correct, Train) ID={c1_id}',
         train_raw[train_raw['SK_ID_CURR'] == c1_id].iloc[0], 0),
        ('client2_wrong_train.pdf',
         f'Client 2 (Wrong, Train) ID={c2_id}',
         train_raw[train_raw['SK_ID_CURR'] == c2_id].iloc[0], 1),
        ('client_test.pdf',
         f'Client 3 (Test) ID={c3_id}',
         test_raw[test_raw['SK_ID_CURR'] == c3_id].iloc[0], 2),
    ]

    for filename, label, raw_row, si in configs:
        pdf_path = os.path.join(OUTPUT_DIR, filename)
        print(f"\nGenerating {filename}...")
        with PdfPages(pdf_path) as pdf:
            plot_client_profile(raw_row, train_raw, scores[si], label, pdf)
            plot_shap_waterfall(
                explainer, sv[si], ev, clients_X.iloc[si],
                f'{label}\nSHAP Waterfall – Score = {scores[si]:.4f}', pdf,
            )
        print(f"  Saved: {pdf_path}")

    test_preds = model.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({'SK_ID_CURR': ids_test, 'TARGET': test_preds})
    sub_path = os.path.join(BASE_DIR, 'results', 'submission.csv')
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    print("Done.")
