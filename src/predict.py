"""
predict.py — Script de prédiction en production
Projet : Analyse Comportementale Clientèle Retail

Usage :
    python src/predict.py --mode churn --client data/test_client_fidele.json
    python src/predict.py --mode churn --input data/train_test/X_test.csv
"""

import os, sys, json, argparse, warnings
import joblib, numpy as np, pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────────
def load_artifacts():
    try:
        model    = joblib.load(os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
        imputer  = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        return model, imputer, scaler, features
    except FileNotFoundError as e:
        print(f"[ERREUR] Artefact manquant : {e}")
        print("         Lancez d'abord : python src/train_model.py")
        sys.exit(1)

def load_artifacts_rfm():
    try:
        kmeans     = joblib.load(os.path.join(MODELS_DIR, 'kmeans_rfm.pkl'))
        scaler_rfm = joblib.load(os.path.join(MODELS_DIR, 'scaler_rfm.pkl'))
        return kmeans, scaler_rfm
    except FileNotFoundError as e:
        print(f"[ERREUR] Artefact RFM manquant : {e}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────
# PREPROCESSING — aligne les colonnes sur ce qu'attend le modèle
# Les colonnes absentes sont remplies par NaN → imputées par médiane
# ─────────────────────────────────────────────────────────────
def align_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Aligne df sur la liste exacte des features du modèle.
    - Colonnes manquantes → NaN  (l'imputer les remplace par la médiane du train)
    - Colonnes en trop    → supprimées
    - Ordre des colonnes  → identique à l'entraînement
    """
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    return df[features]


def preprocess(df: pd.DataFrame, imputer, scaler, features: list) -> pd.DataFrame:
    df = align_features(df.copy(), features)
    df = pd.DataFrame(imputer.transform(df),  columns=features)
    df = pd.DataFrame(scaler.transform(df),   columns=features)
    return df


# ─────────────────────────────────────────────────────────────
# CONVERSION DES CHAMPS FORMULAIRE → FEATURES MODÈLE
# Traduit les 8 champs simples du formulaire Flask en colonnes
# que le modèle comprend (one-hot, ordinal, etc.)
# ─────────────────────────────────────────────────────────────
#
# Dans `src/preprocessing.py`, `SpendingCategory` est encodée ordinalement via
# `pd.Categorical(..., categories=['Low','Medium','High','VIP']).codes`.
# Donc les codes attendus sont :
# Low=0, Medium=1, High=2, VIP=3.
#
SPENDING_MAP  = {'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3}
SEASON_COLS   = ['FavoriteSeason_Hiver', 'FavoriteSeason_Printemps', 'FavoriteSeason_Été']
SEASON_MAP    = {'Hiver': 0, 'Printemps': 1, 'Été': 2, 'Automne': 3}

def form_to_features(form: dict) -> dict:
    """
    Convertit les 8 champs du formulaire Flask en dict de features.
    Toutes les autres features seront NaN → imputées par médiane.
    """
    d = {}

    # Champs directs
    d['Frequency']               = float(form.get('Frequency', 3))
    d['MonetaryTotal']           = float(form.get('MonetaryTotal', 500))
    d['AvgDaysBetweenPurchases'] = float(form.get('AvgDaysBetweenPurchases', 30))
    d['UniqueProducts']          = float(form.get('UniqueProducts', 5))
    d['PreferredMonth']          = float(form.get('PreferredMonth', 6))
    d['RegYear']                 = float(form.get('RegYear', 2012))

    # SpendingCategory : texte (UI) ou valeur numérique (JSON CLI).
    spending_raw = form.get('SpendingCategory', 'Medium')
    if isinstance(spending_raw, (int, float)) and not isinstance(spending_raw, bool):
        # Déjà encodé (0..3)
        d['SpendingCategory'] = float(spending_raw)
    else:
        # Valeur texte Low/Medium/High/VIP
        d['SpendingCategory'] = float(SPENDING_MAP.get(spending_raw, 1))

    # FavoriteSeason : texte → one-hot (3 colonnes)
    season_raw = form.get('FavoriteSeason', 'Automne')
    season_idx = SEASON_MAP.get(season_raw, 3)
    for i, col in enumerate(SEASON_COLS):
        d[col] = 1.0 if i == season_idx else 0.0

    # AvgBasketValue dérivé
    freq = d['Frequency'] if d['Frequency'] > 0 else 1
    d['AvgBasketValue'] = d['MonetaryTotal'] / freq

    return d


# ─────────────────────────────────────────────────────────────
# PRÉDICTION CHURN
# ─────────────────────────────────────────────────────────────
SEGMENT_RFM = {0: 'Clients Champions', 1: 'Clients Fidèles',
               2: 'Clients Perdus',    3: 'Clients à Risque'}

def risk_label(p: float) -> str:
    if p >= 0.75: return 'Critique'
    if p >= 0.50: return 'Élevé'
    if p >= 0.25: return 'Moyen'
    return 'Faible'


def predict_churn(df_input: pd.DataFrame) -> pd.DataFrame:
    model, imputer, scaler, features = load_artifacts()
    customer_ids = df_input['CustomerID'] if 'CustomerID' in df_input.columns \
                   else pd.Series(range(len(df_input)))

    X = preprocess(df_input, imputer, scaler, features)
    churn_pred  = model.predict(X)
    churn_proba = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        'CustomerID'       : customer_ids.values,
        'churn_predicted'  : churn_pred,
        'churn_probability': churn_proba.round(4),
        'risk_segment'     : [risk_label(p) for p in churn_proba],
    })


def predict_churn_from_form(form: dict) -> dict:
    """
    Point d'entrée pour Flask — reçoit les 8 champs du formulaire,
    retourne un dict JSON-serialisable avec tous les résultats.
    """
    model, imputer, scaler, features = load_artifacts()

    # Conversion formulaire → features
    feat_dict = form_to_features(form)
    df = pd.DataFrame([feat_dict])
    X  = preprocess(df, imputer, scaler, features)

    churn_pred  = int(model.predict(X)[0])
    churn_proba = float(model.predict_proba(X)[0, 1])

    # Segment RFM si Frequency + MonetaryTotal disponibles
    rfm_segment = 'N/A'
    try:
        kmeans, scaler_rfm = load_artifacts_rfm()
        rfm_raw = pd.DataFrame([{
            'Recency'      : float(form.get('Recency', 50)),
            'Frequency'    : float(form.get('Frequency', 3)),
            'MonetaryTotal': float(form.get('MonetaryTotal', 500)),
        }])
        rfm_log = rfm_raw.clip(lower=0).apply(np.log1p)
        rfm_sc  = scaler_rfm.transform(rfm_log)
        cluster = int(kmeans.predict(rfm_sc)[0])
        rfm_segment = SEGMENT_RFM.get(cluster, f'Cluster {cluster}')
    except Exception:
        pass

    return {
        'churn_predicted'  : churn_pred,
        'churn_probability': round(churn_proba * 100, 1),
        'risk_segment'     : risk_label(churn_proba),
        'rfm_segment'      : rfm_segment,
    }


def predict_churn_with_rfm(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Variante CLI qui ajoute aussi un segment RFM (si Recency/Frequency/MonetaryTotal existent).
    """
    results = predict_churn(df_input)
    rfm_segment = 'N/A'

    try:
        kmeans, scaler_rfm = load_artifacts_rfm()

        if {'Recency', 'Frequency', 'MonetaryTotal'}.issubset(df_input.columns):
            rfm_raw = pd.DataFrame([{
                'Recency': float(df_input['Recency'].iloc[0]),
                'Frequency': float(df_input['Frequency'].iloc[0]),
                'MonetaryTotal': float(df_input['MonetaryTotal'].iloc[0]),
            }])

            rfm_log = rfm_raw.clip(lower=0).apply(np.log1p)
            rfm_sc = scaler_rfm.transform(rfm_log)
            cluster = int(kmeans.predict(rfm_sc)[0])
            rfm_segment = SEGMENT_RFM.get(cluster, f'Cluster {cluster}')
    except Exception:
        pass

    results['rfm_segment'] = rfm_segment
    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Prédiction Churn Retail ML')
    parser.add_argument('--mode',   choices=['churn', 'all'], default='churn')
    parser.add_argument('--input',  type=str, default=None)
    parser.add_argument('--client', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.client:
        # Supporte fichier .json ET string JSON inline
        if args.client.endswith('.json'):
            with open(args.client, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(args.client)
        df_input = pd.DataFrame([data])
        print(f"[INFO] Prédiction pour 1 client fourni en JSON")

    elif args.input:
        df_input = pd.read_csv(args.input)
        print(f"[INFO] Données chargées : {df_input.shape}")
    else:
        print("[ERREUR] Fournissez --input <csv> ou --client <json>")
        sys.exit(1)

    if args.mode == 'all':
        results = predict_churn_with_rfm(df_input)
    else:
        results = predict_churn(df_input)
    out_path = args.output or os.path.join(OUT_DIR, 'predictions_churn.csv')

    print("\n-- Resultats ----------------------------------------")
    print(results.to_string(index=False))
    results.to_csv(out_path, index=False)
    print(f"\n[OK] Prédictions sauvegardées : {out_path}")