import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.final_cleaning import final_clean

def test_final_clean():
    df = pd.DataFrame({
        "Nom@Colonne!": [1, 2, None],
        "Autre#Colonne$": [None, 4, 5],
        "Colonne OK": [5, None, 6],
        "A": [1, None, None],
        "B": [None, None, None]
    })

    df_cleaned = final_clean(df, cols_fill_zero=["Nom@Colonne!"])
    
    # Vérifie que les colonnes ont été renommées correctement
    for col in df_cleaned.columns:
        assert all(c.isalnum() or c == "_" for c in col)
    
    # Vérifie que les NaN ont été remplis
    assert df_cleaned.isna().sum().sum() == 0
 


