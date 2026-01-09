import sys
import os
import streamlit as st
import pandas as pd

# Répertoire de base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.predict import ModelPredictor

# Chemins vers le modèle, les features et les 5 clients
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_lightgbm.pkl")
TOP_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "top_features.csv")
CLIENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "five_clients.csv")

# Charger le modèle et les features
predictor = ModelPredictor(
    model_path=MODEL_PATH,
    top_features_path=TOP_FEATURES_PATH,
    threshold=0.45  # seuil métier
)

# Charger les 5 clients
clients_df = pd.read_csv(CLIENTS_PATH)

st.title("Application de scoring crédit")
st.write("Sélectionnez un client pour prédire le risque de défaut")

# Menu déroulant pour choisir le client
client_id = st.selectbox(
    "Choisir un client",
    options=clients_df.index.tolist()
)

# Récupérer les données du client sélectionné
client_data = clients_df.loc[[client_id]]
st.subheader("Features du client")
st.dataframe(client_data)

# Bouton de prédiction
if st.button("Prédire"):
    try:
        proba = predictor.predict_proba(client_data)[0]
        classe = predictor.predict_class(client_data)[0]

        st.success(f"Probabilité de défaut : {proba:.2f}")
        st.info(f"Classe prédite : {'Défaillant' if classe == 1 else 'Accordé'}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

