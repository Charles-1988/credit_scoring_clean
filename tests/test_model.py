import os
import pandas as pd
from src.predict import ModelPredictor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_lightgbm.pkl")
TOP_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "top_features.csv")

# Instanciation directe du modèle
predictor = ModelPredictor(
    model_path=MODEL_PATH,
    top_features_path=TOP_FEATURES_PATH,
    threshold=0.45
)

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)

def test_top_features_file_exists():
    assert os.path.exists(TOP_FEATURES_PATH)
    df = pd.read_csv(TOP_FEATURES_PATH)
    assert "feature" in df.columns

def test_predict_proba():
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    probas = predictor.predict_proba(test_df)
    assert all(0 <= p <= 1 for p in probas)

def test_predict_class():
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    classes = predictor.predict_class(test_df)
    assert all(c in [0, 1] for c in classes)

