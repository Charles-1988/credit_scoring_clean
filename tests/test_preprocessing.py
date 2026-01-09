import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    application_features, one_hot_encoder, bureau_features
)

@pytest.fixture
def df_application():
    return pd.DataFrame({
        "DAYS_BIRTH": [10000, 20000],
        "DAYS_EMPLOYED": [1000, 365243],
        "AMT_INCOME_TOTAL": [50000, 60000],
        "AMT_CREDIT": [100000, 150000],
        "CNT_FAM_MEMBERS": [3, 4],
        "AMT_ANNUITY": [2000, 3000]
    })

@pytest.fixture
def df_bureau():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "SK_ID_BUREAU": [101, 102],
        "DAYS_CREDIT": [100, 200],
        "AMT_CREDIT_SUM": [5000, 6000],
        "AMT_ANNUITY": [100, 200],
        "CREDIT_ACTIVE": ["Active", "Closed"]
    })

@pytest.fixture
def df_bureau_balance():
    return pd.DataFrame({
        "SK_ID_BUREAU": [101, 102],
        "MONTHS_BALANCE": [1, 2],
        "STATUS": ["0", "1"]
    })

def test_application_features(df_application):
    df_feat = application_features(df_application)
    assert "DAYS_EMPLOYED_PERC" in df_feat.columns
    assert "INCOME_CREDIT_PERC" in df_feat.columns

def test_one_hot_encoder(df_bureau):
    df_encoded, new_cols = one_hot_encoder(df_bureau)
    assert all(col.startswith("CREDIT_ACTIVE") or col in df_bureau.columns for col in df_encoded.columns)

def test_bureau_features(df_bureau, df_bureau_balance):
    bureau_agg = bureau_features(df_bureau, df_bureau_balance)
    assert "BURO_DAYS_CREDIT_MIN" in bureau_agg.columns


