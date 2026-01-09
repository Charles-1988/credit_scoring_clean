from fastapi import FastAPI
import pandas as pd
from src.predict import ModelPredictor
import os

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_lightgbm.pkl")
TOP_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "top_features.csv")

predictor = ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, threshold=0.45)

@app.get("/")
def read_root():
    return {"message": "API Credit Scoring"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    return {"proba": float(proba), "classe": int(classe)}


