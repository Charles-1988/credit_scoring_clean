import pandas as pd
import numpy as np
from src.preprocessing import clean_column_names

def replace_inf(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def fill_missing_with_zero(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    return df

def fill_missing_with_median(df, exclude_cols=[]):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    for col in numeric_cols:
        if col not in exclude_cols:
            median_val = df[col].median()
            df[col + '_is_missing'] = df[col].isna().astype(int)
            df[col].fillna(median_val, inplace=True)
    return df

def drop_high_na_columns(df, threshold=0.7):
    df = df.copy()
    cols_to_drop = df.columns[df.isna().mean() > threshold]
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def final_clean(df, cols_fill_zero=[]):
    df = df.copy()
    df = replace_inf(df)
    df = fill_missing_with_zero(df, cols_fill_zero)
    df = fill_missing_with_median(df, exclude_cols=cols_fill_zero)
    df = drop_high_na_columns(df, threshold=0.7)
    df = clean_column_names(df)
    return df

