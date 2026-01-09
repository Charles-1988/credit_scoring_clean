import numpy as np
import pandas as pd

def one_hot_encoder(df, nan_as_category=True):
    """Encode les colonnes catégorielles"""
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def clean_column_names(df):
    """Supprime les caractères spéciaux dans les noms de colonnes"""
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

def application_features(df):
    """Crée des ratios et features sur application principale"""
    df = df.copy()
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    epsilon = 1e-6  # éviter division par zéro
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + epsilon)
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + epsilon)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + epsilon)
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + epsilon)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + epsilon)
    return df

def bureau_features(bureau, bureau_balance):
    bureau = bureau.copy()
    bureau_balance = bureau_balance.copy()
    
    bureau, bureau_cat = one_hot_encoder(bureau)
    bureau_balance, bb_cat = one_hot_encoder(bureau_balance)

    # Aggregation bureau_balance
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        **{col: ['mean'] for col in bb_cat}
    })
    bb_agg.columns = ['BB_' + '_'.join(col).upper() for col in bb_agg.columns]
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    # Aggregation par client
    num_agg = {
        'DAYS_CREDIT': ['min','max','mean','var'],
        'AMT_CREDIT_SUM': ['max','mean','sum'],
        'AMT_ANNUITY': ['max','mean']
    }
    cat_agg = {col: ['mean'] for col in bureau_cat}
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_agg, **cat_agg})
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]
    return bureau_agg

def previous_app_features(prev):
    prev = prev.copy()
    prev, cat_cols = one_hot_encoder(prev)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1e-6)

    num_agg = {
        'AMT_ANNUITY': ['min','max','mean'],
        'AMT_APPLICATION': ['min','max','mean'],
        'AMT_CREDIT': ['min','max','mean'],
        'APP_CREDIT_PERC': ['min','max','mean','var']
    }
    cat_agg = {col: ['mean'] for col in cat_cols}

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_agg, **cat_agg})
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    return prev_agg

def pos_features(pos):
    pos = pos.copy()
    pos, cat_cols = one_hot_encoder(pos)
    num_agg = {
        'MONTHS_BALANCE': ['max','mean','size'],
        'SK_DPD': ['max','mean'],
        'SK_DPD_DEF': ['max','mean']
    }
    cat_agg = {col: ['mean'] for col in cat_cols}
    pos_agg = pos.groupby('SK_ID_CURR').agg({**num_agg, **cat_agg})
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    return pos_agg

def instal_features(df):
    df = df.copy()
    df, cat_cols = one_hot_encoder(df)
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 1e-6)
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)

    ins_agg = df.groupby('SK_ID_CURR').agg({
        'DPD': ['max','mean','sum'],
        'PAYMENT_PERC': ['max','mean','sum','var'],
        'PAYMENT_DIFF': ['max','mean','sum','var']
    })
    ins_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in ins_agg.columns]
    return ins_agg

def cc_features(df):
    df = df.copy()
    df, cat_cols = one_hot_encoder(df)
    cc_agg = df.groupby('SK_ID_CURR').agg(['min','max','mean','sum','var'])
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    return cc_agg

