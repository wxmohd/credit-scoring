import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def load_data():
    """Load all datasets from the data directory."""
    print("Loading application data...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'application_train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'application_test.csv'))

    print("Loading bureau data...")
    bureau         = pd.read_csv(os.path.join(DATA_DIR, 'bureau.csv'))
    bureau_balance = pd.read_csv(os.path.join(DATA_DIR, 'bureau_balance.csv'))

    print("Loading previous application data...")
    prev = pd.read_csv(os.path.join(DATA_DIR, 'previous_application.csv'))

    print("Loading POS cash data...")
    pos = pd.read_csv(os.path.join(DATA_DIR, 'POS_CASH_balance.csv'))

    print("Loading credit card data...")
    cc = pd.read_csv(os.path.join(DATA_DIR, 'credit_card_balance.csv'))

    print("Loading installments data...")
    installments = pd.read_csv(os.path.join(DATA_DIR, 'installments_payments.csv'))

    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test, bureau, bureau_balance, prev, pos, cc, installments


def aggregate_bureau_balance(bureau_balance):
    """Aggregate bureau_balance by SK_ID_BUREAU."""
    bb = bureau_balance.copy()
    bb['STATUS_DPD'] = bb['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(
        BB_MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
        BB_MONTHS_BALANCE_MIN=('MONTHS_BALANCE', 'min'),
        BB_MONTHS_BALANCE_COUNT=('MONTHS_BALANCE', 'count'),
        BB_STATUS_DPD_SUM=('STATUS_DPD', 'sum'),
        BB_STATUS_DPD_MEAN=('STATUS_DPD', 'mean'),
    ).reset_index()
    return bb_agg


def aggregate_bureau(bureau, bureau_balance):
    """Aggregate bureau and bureau_balance by SK_ID_CURR."""
    bb_agg = aggregate_bureau_balance(bureau_balance)
    bur = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

    bureau_agg = bur.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_AMT_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
        BUREAU_AMT_CREDIT_MEAN=('AMT_CREDIT_SUM', 'mean'),
        BUREAU_AMT_CREDIT_MAX=('AMT_CREDIT_SUM', 'max'),
        BUREAU_AMT_CREDIT_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_AMT_CREDIT_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
        BUREAU_DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'),
        BUREAU_DAYS_CREDIT_MIN=('DAYS_CREDIT', 'min'),
        BUREAU_DAYS_CREDIT_MAX=('DAYS_CREDIT', 'max'),
        BUREAU_DAYS_CREDIT_UPDATE_MEAN=('DAYS_CREDIT_UPDATE', 'mean'),
        BUREAU_CNT_CREDIT_PROLONG_SUM=('CNT_CREDIT_PROLONG', 'sum'),
        BUREAU_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        BUREAU_ACTIVE_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        BUREAU_CLOSED_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        BUREAU_BB_DPD_MEAN=('BB_STATUS_DPD_MEAN', 'mean'),
        BUREAU_BB_DPD_SUM=('BB_STATUS_DPD_SUM', 'sum'),
    ).reset_index()

    bureau_agg['BUREAU_ACTIVE_RATIO'] = (
        bureau_agg['BUREAU_ACTIVE_COUNT'] / (bureau_agg['BUREAU_LOAN_COUNT'] + 1)
    )
    bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_DEBT_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM'] + 1)
    )
    return bureau_agg


def aggregate_previous_application(prev):
    """Aggregate previous applications by SK_ID_CURR."""
    prev = prev.copy()
    prev['APP_CREDIT_RATIO']   = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1)
    prev['CREDIT_GOODS_RATIO'] = prev['AMT_CREDIT'] / (prev['AMT_GOODS_PRICE'] + 1)
    prev['DAYS_DECISION']      = prev['DAYS_DECISION'].replace(365243, np.nan)

    prev_agg = prev.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        PREV_AMT_CREDIT_SUM=('AMT_CREDIT', 'sum'),
        PREV_AMT_CREDIT_MAX=('AMT_CREDIT', 'max'),
        PREV_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        PREV_AMT_GOODS_PRICE_MEAN=('AMT_GOODS_PRICE', 'mean'),
        PREV_DAYS_DECISION_MEAN=('DAYS_DECISION', 'mean'),
        PREV_DAYS_DECISION_MIN=('DAYS_DECISION', 'min'),
        PREV_APP_CREDIT_RATIO_MEAN=('APP_CREDIT_RATIO', 'mean'),
        PREV_CREDIT_GOODS_RATIO_MEAN=('CREDIT_GOODS_RATIO', 'mean'),
        PREV_APPROVED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        PREV_REFUSED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        PREV_CANCELED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Canceled').sum()),
        PREV_CONSUMER_COUNT=('NAME_CONTRACT_TYPE', lambda x: (x == 'Consumer loans').sum()),
        PREV_CASH_COUNT=('NAME_CONTRACT_TYPE', lambda x: (x == 'Cash loans').sum()),
        PREV_CNT_PAYMENT_MEAN=('CNT_PAYMENT', 'mean'),
        PREV_DAYS_FIRST_DUE_MEAN=('DAYS_FIRST_DUE', 'mean'),
    ).reset_index()

    prev_agg['PREV_REFUSED_RATIO']   = prev_agg['PREV_REFUSED_COUNT']   / (prev_agg['PREV_APP_COUNT'] + 1)
    prev_agg['PREV_APPROVED_RATIO']  = prev_agg['PREV_APPROVED_COUNT']  / (prev_agg['PREV_APP_COUNT'] + 1)
    return prev_agg


def aggregate_pos_cash(pos):
    """Aggregate POS_CASH_balance by SK_ID_CURR."""
    pos_agg = pos.groupby('SK_ID_CURR').agg(
        POS_MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
        POS_MONTHS_BALANCE_MIN=('MONTHS_BALANCE', 'min'),
        POS_MONTHS_BALANCE_SIZE=('MONTHS_BALANCE', 'size'),
        POS_SK_DPD_MEAN=('SK_DPD', 'mean'),
        POS_SK_DPD_MAX=('SK_DPD', 'max'),
        POS_SK_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        POS_SK_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
        POS_ACTIVE_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
        POS_COMPLETED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
    ).reset_index()
    return pos_agg


def aggregate_credit_card(cc):
    """Aggregate credit_card_balance by SK_ID_CURR."""
    cc = cc.copy()
    cc['BALANCE_LIMIT_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    cc['DRAWING_RATIO']       = cc['AMT_DRAWINGS_CURRENT'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

    cc_agg = cc.groupby('SK_ID_CURR').agg(
        CC_MONTHS_BALANCE_COUNT=('MONTHS_BALANCE', 'count'),
        CC_AMT_BALANCE_MEAN=('AMT_BALANCE', 'mean'),
        CC_AMT_BALANCE_MAX=('AMT_BALANCE', 'max'),
        CC_AMT_CREDIT_LIMIT_MEAN=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_AMT_DRAWINGS_MEAN=('AMT_DRAWINGS_CURRENT', 'mean'),
        CC_AMT_PAYMENT_MEAN=('AMT_PAYMENT_CURRENT', 'mean'),
        CC_SK_DPD_MEAN=('SK_DPD', 'mean'),
        CC_SK_DPD_MAX=('SK_DPD', 'max'),
        CC_BALANCE_LIMIT_RATIO_MEAN=('BALANCE_LIMIT_RATIO', 'mean'),
        CC_DRAWING_RATIO_MEAN=('DRAWING_RATIO', 'mean'),
        CC_CNT_DRAWINGS_MEAN=('CNT_DRAWINGS_CURRENT', 'mean'),
    ).reset_index()
    return cc_agg


def aggregate_installments(installments):
    """Aggregate installments_payments by SK_ID_CURR."""
    inst = installments.copy()
    inst['PAYMENT_DIFF']  = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['DAYS_PAST_DUE'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
    inst['LATE_PAYMENT']  = (inst['DAYS_PAST_DUE'] > 0).astype(int)
    inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1)

    inst_agg = inst.groupby('SK_ID_CURR').agg(
        INST_COUNT=('SK_ID_PREV', 'count'),
        INST_AMT_INSTALMENT_MEAN=('AMT_INSTALMENT', 'mean'),
        INST_AMT_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
        INST_AMT_PAYMENT_MEAN=('AMT_PAYMENT', 'mean'),
        INST_AMT_PAYMENT_SUM=('AMT_PAYMENT', 'sum'),
        INST_PAYMENT_DIFF_MEAN=('PAYMENT_DIFF', 'mean'),
        INST_PAYMENT_DIFF_MAX=('PAYMENT_DIFF', 'max'),
        INST_PAYMENT_DIFF_SUM=('PAYMENT_DIFF', 'sum'),
        INST_DAYS_PAST_DUE_MEAN=('DAYS_PAST_DUE', 'mean'),
        INST_DAYS_PAST_DUE_MAX=('DAYS_PAST_DUE', 'max'),
        INST_LATE_PAYMENT_MEAN=('LATE_PAYMENT', 'mean'),
        INST_LATE_PAYMENT_SUM=('LATE_PAYMENT', 'sum'),
        INST_PAYMENT_RATIO_MEAN=('PAYMENT_RATIO', 'mean'),
    ).reset_index()
    return inst_agg


def engineer_application_features(df):
    """Create domain-specific features from application data."""
    df = df.copy()

    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']  / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT']   / (df['AMT_ANNUITY']      + 1)
    df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT']    + 1)

    df['AGE_YEARS']          = -df['DAYS_BIRTH']    / 365
    df['EMPLOYMENT_YEARS']   = -df['DAYS_EMPLOYED'] / 365
    df['DAYS_EMPLOYED_RATIO']= df['DAYS_EMPLOYED']  / (df['DAYS_BIRTH'] + 1)
    df['INCOME_PER_PERSON']  = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext].mean(axis=1)
    df['EXT_SOURCE_STD']  = df[ext].std(axis=1)
    df['EXT_SOURCE_MIN']  = df[ext].min(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_2']  = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3']  = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3']  = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)

    df['DAYS_ID_PUBLISH_REG_DIFF'] = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
    return df


def encode_categoricals(df):
    """Label-encode all object columns."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.Categorical(df[col]).codes.astype(np.int16)
    return df


def build_features(train, test, bureau, bureau_balance, prev, pos, cc, installments):
    """Build complete feature matrix by merging all supplementary tables."""
    print("Aggregating bureau features...")
    bureau_feats = aggregate_bureau(bureau, bureau_balance)

    print("Aggregating previous application features...")
    prev_feats = aggregate_previous_application(prev)

    print("Aggregating POS cash features...")
    pos_feats = aggregate_pos_cash(pos)

    print("Aggregating credit card features...")
    cc_feats = aggregate_credit_card(cc)

    print("Aggregating installment features...")
    inst_feats = aggregate_installments(installments)

    def merge_all(df):
        df = engineer_application_features(df)
        df = df.merge(bureau_feats, on='SK_ID_CURR', how='left')
        df = df.merge(prev_feats,   on='SK_ID_CURR', how='left')
        df = df.merge(pos_feats,    on='SK_ID_CURR', how='left')
        df = df.merge(cc_feats,     on='SK_ID_CURR', how='left')
        df = df.merge(inst_feats,   on='SK_ID_CURR', how='left')
        df = encode_categoricals(df)
        return df

    print("Building train feature matrix...")
    train_feat = merge_all(train)
    print("Building test feature matrix...")
    test_feat  = merge_all(test)
    return train_feat, test_feat


if __name__ == "__main__":
    train, test, bureau, bureau_balance, prev, pos, cc, installments = load_data()
    train_feat, test_feat = build_features(
        train, test, bureau, bureau_balance, prev, pos, cc, installments
    )
    print(f"Train features: {train_feat.shape}")
    print(f"Test features:  {test_feat.shape}")
    print("Preprocessing complete.")
