import joblib
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, model_path="models/best_model_lightgbm.pkl",
                 top_features_path="../data/processed/top_features.csv",
                 threshold=0.5):
        self.model = joblib.load(model_path)
        self.top_features = pd.read_csv(top_features_path)["feature"].tolist()
        self.threshold = threshold

    def predict_proba(self, X):
        X_sel = X[self.top_features]
        proba = self.model.predict_proba(X_sel)[:,1]
        return proba

    def predict_class(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def set_threshold(self, threshold):
        self.threshold = threshold
