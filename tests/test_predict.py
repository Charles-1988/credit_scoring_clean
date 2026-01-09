import sys
import os
import pytest
import pandas as pd
import numpy as np

# Permet d'importer src même si on lance pytest depuis tests/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import ModelPredictor

@pytest.fixture
def model():
    # Crée un objet ModelPredictor
    return ModelPredictor(
        model_path=os.path.join("models", "best_model_lightgbm.pkl"),
        top_features_path=os.path.join("data", "processed", "top_features.csv"),
        threshold=0.5
    )

@pytest.fixture
def X_sample(model):
    """
    Crée un DataFrame factice avec toutes les colonnes nécessaires
    (top_features) pour que le modèle puisse prédire sans KeyError.
    """
    n_samples = 3
    # Remplit toutes les colonnes par 0.5 ou toute valeur arbitraire
    data = {col: [0.5]*n_samples for col in model.top_features}
    return pd.DataFrame(data)

def test_predict_shapes(model, X_sample):
    # Test des probabilités
    proba = model.predict_proba(X_sample)
    assert proba.shape[0] == X_sample.shape[0]
    assert np.all((proba >= 0) & (proba <= 1))

    # Test des classes
    y_pred = model.predict_class(X_sample)
    assert y_pred.shape[0] == X_sample.shape[0]
    assert set(np.unique(y_pred)).issubset({0, 1})








