"""
predict.py — Prédiction en production
Projet : Analyse Comportementale Clientèle Retail

Usage :
    python src/predict.py --mode churn  --client data/test_client.json
    python src/predict.py --mode all    --input  data/train_test/X_test.csv
"""

import os, sys, json, argparse, warnings
import joblib
import numpy  as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

ALL_SEGMENTS = ['Clients Champions', 'Clients Fideles',
                'Clients Perdus',    'Clients a Risque']

# ── Mappings d'encodage (cohérents avec preprocessing.py) ────────────────────
SPENDING_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3}

RFM_SEGMENT_MAP = {
    'Dormants': 0,
    'Potentiels': 1,
    'Fidèles': 2,
    'Champions': 3
}

SEASON_COLS = ['FavoriteSeason_Hiver', 'FavoriteSeason_Printemps', 'FavoriteSeason_Été']
SEASON_MAP  = {'Hiver': 0, 'Printemps': 1, 'Ete': 2, 'Été': 2, 'Automne': 3}


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────────

def load_artifacts():
    try:
        model    = joblib.load(os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
        imputer  = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        return model, imputer, features
    except FileNotFoundError as e:
        print(f"[ERREUR] Artefact manquant : {e}")
        print("         Lancez d'abord : python src/train_model.py")
        sys.exit(1)

def load_artifacts_regression():
    """Charge le modèle de régression et ses métadonnées."""
    try:
        model    = joblib.load(os.path.join(MODELS_DIR, 'regressor_monetary.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'regression_metadata.pkl'))
        return model, metadata
    except FileNotFoundError as e:
        print(f"[WARN] Artefact régression manquant : {e}")
        return None, None
    
def load_artifacts_rfm():
    """Artefacts du clustering K-Means (usage marketing uniquement)."""
    try:
        kmeans      = joblib.load(os.path.join(MODELS_DIR, 'kmeans_rfm.pkl'))
        scaler_rfm  = joblib.load(os.path.join(MODELS_DIR, 'scaler_rfm.pkl'))
        imputer_rfm = joblib.load(os.path.join(MODELS_DIR, 'imputer_rfm.pkl'))
        seg_map     = joblib.load(os.path.join(MODELS_DIR, 'cluster_segment_map.pkl'))
        return kmeans, scaler_rfm, imputer_rfm, seg_map
    except FileNotFoundError as e:
        print(f"[ERREUR] Artefact RFM manquant : {e}")
        sys.exit(1)


def load_country_encoding():
    try:
        return joblib.load(os.path.join(MODELS_DIR, 'country_encoding.pkl'))
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def align_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Aligne le DataFrame sur la liste exacte des features du modèle.
    - Colonnes manquantes → NaN  (imputées par médiane du train)
    - Colonnes en trop    → supprimées
    - Ordre               → identique à l'entraînement
    """
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    return df[features]


def preprocess(df: pd.DataFrame, imputer, features: list) -> pd.DataFrame:
    """Aligne + impute. Pas de scaling (non utilisé pour le RF en classification)."""
    df = align_features(df.copy(), features)
    df = pd.DataFrame(imputer.transform(df), columns=features)
    return df

# CLUSTER RFM (usage marketing uniquement — PAS une feature du modèle)

def calculer_rfm_cluster(recency: float, frequency: float, monetary: float):
    """
    Calcule le segment RFM via le clustering K-Means entraîné.
    C'est une INFO MARKETING d'affichage.
    N'est PAS injecté dans les features de classification.
    """
    try:
        kmeans, scaler_rfm, imputer_rfm, seg_map = load_artifacts_rfm()

        rfm_raw = pd.DataFrame([{
            'Recency'      : float(recency),
            'Frequency'    : float(frequency),
            'MonetaryTotal': float(monetary),
        }])

        # Pipeline identique à train_model.py : imputer → clip → log → scaler
        rfm_imputed = pd.DataFrame(
            imputer_rfm.transform(rfm_raw),
            columns=['Recency', 'Frequency', 'MonetaryTotal']
        )
        rfm_log = rfm_imputed.clip(lower=0).apply(np.log1p)
        rfm_sc  = scaler_rfm.transform(rfm_log)

        cluster     = int(kmeans.predict(rfm_sc)[0])
        rfm_segment = seg_map.get(cluster, f'Cluster {cluster}')
        return cluster, rfm_segment

    except Exception as e:
        print(f"[WARN] RFM cluster échoué : {e}")
        return -1, 'N/A'

def predict_monetary_from_form(form: dict) -> dict:
    """
    Prédit MonetaryTotal via le régresseur XGBoost/LightGBM/RF entraîné.
    Utilise le même pipeline que la classification :
        align_features → imputation → prédiction → expm1 (inverse log)
    """
    model_reg, metadata = load_artifacts_regression()
    if model_reg is None:
        return {'monetary_predicted': None, 'monetary_model': 'N/A'}

    _, imputer, _ = load_artifacts()

    # Les features de la régression = même X_train que classification
    feature_cols = metadata.get('feature_columns', [])

    # Construire le dict de features depuis le formulaire
    feat_dict = form_to_features(form)
    df_input  = pd.DataFrame([feat_dict])

    # Aligner + imputer sur les features de la régression
    df_aligned = align_features(df_input.copy(), feature_cols)
    df_imputed = pd.DataFrame(
        imputer.transform(df_aligned),
        columns=feature_cols
    )

    # Prédiction en espace log puis inverse
    y_log  = model_reg.predict(df_imputed)
    y_pred = float(np.expm1(y_log)[0])
    y_pred = max(0.0, round(y_pred, 2))  # pas de valeur négative

    return {
        'monetary_predicted': y_pred,
        'monetary_model'    : metadata.get('model_type', 'unknown'),
        'monetary_mape'     : metadata.get('mape_test',  None),
    }
    
# CONVERSION FORMULAIRE → FEATURES

def form_to_features(form: dict) -> dict:
    d = {}

    # ── Features numériques directes ─────────────────────────
    d['Frequency']               = float(form.get('Frequency', 3))
    d['AvgDaysBetweenPurchases'] = float(form.get('AvgDaysBetweenPurchases', 30))
    d['UniqueProducts']          = float(form.get('UniqueProducts', 5))
    d['PreferredMonth']          = float(form.get('PreferredMonth', 6))
    # d['CustomerTenureDays']      = float(form.get('CustomerTenureDays', 365))
    d['FirstPurchaseDaysAgo']    = float(form.get('FirstPurchaseDaysAgo', 360))
    d['MonetartyTotal']          = float(form.get('MonetartyTotal', 360))


    # SUPPRIMÉES : Recency (clustering only), RFMSegment (droppée), 
    #              MonetaryTotal (target régression — à vérifier)

    # ── SpendingCategory ──────────────────────────────────────
    spending_raw = form.get('SpendingCategory', 'Medium')
    if isinstance(spending_raw, (int, float)) and not isinstance(spending_raw, bool):
        d['SpendingCategory'] = float(spending_raw)
    else:
        d['SpendingCategory'] = float(SPENDING_MAP.get(str(spending_raw), 1))

    # ── AvgBasketValue (dérivée) ──────────────────────────────
    freq = d['Frequency'] if d['Frequency'] > 0 else 1
    monetary = float(form.get('MonetaryTotal', 500))
    d['AvgBasketValue'] = round(monetary / freq, 4)

    # ── Date d'inscription ────────────────────────────────────
    d['RegYear']    = np.nan
    d['RegMonth']   = np.nan
    d['RegDay']     = np.nan
    d['RegWeekday'] = np.nan

    reg_date = form.get('RegistrationDate', None)
    reg_year = form.get('RegYear', None)

    if reg_date:
        dt = pd.to_datetime(reg_date, dayfirst=True, errors='coerce')
        if pd.notna(dt):
            d['RegYear']    = float(dt.year)
            d['RegMonth']   = float(dt.month)
            d['RegDay']     = float(dt.day)
            d['RegWeekday'] = float(dt.weekday())
    if pd.isna(d['RegYear']) and reg_year is not None:
        d['RegYear'] = float(reg_year)

    # ── FavoriteSeason (one-hot) ──────────────────────────────
    season_raw = form.get('FavoriteSeason', 'Automne')
    season_idx = SEASON_MAP.get(str(season_raw), 3)
    for i, col in enumerate(SEASON_COLS):
        d[col] = 1.0 if i == season_idx else 0.0

    # ── Country_encoded ───────────────────────────────────────
    country = form.get('Country', None)
    if country:
        enc = load_country_encoding()
        if enc:
            d['Country_encoded'] = enc['mapping'].get(country, enc['moy_globale'])

    return d

# LABEL DE RISQUE

def risk_label(p: float) -> str:
    if p >= 0.75: return 'Critique'
    if p >= 0.50: return 'Eleve'
    if p >= 0.25: return 'Moyen'
    return 'Faible'


def marketing_recommendations(rfm_segment: str, churn_proba: float) -> dict:
    """Recommandations marketing selon le segment RFM (clustering)."""
    base = {
        'Clients Champions': {
            'priorite': 'Fidélisation premium',
            'couleur': 'green',
            'actions': [
                "Inviter au programme VIP exclusif",
                "Proposer un early access sur les nouveaux produits",
            ]
        },
        'Clients Fideles': {
            'priorite': 'Engagement & croissance',
            'couleur': 'teal',
            'actions': [
                "Lancer une campagne de parrainage avec récompense",
                "Cross-sell basé sur l'historique d'achat",
            ]
        },
        'Clients a Risque': {
            'priorite': 'Réengagement urgent',
            'couleur': 'amber',
            'actions': [
                "Envoyer un bon de réduction ciblé sous 48h",
                "Email de réengagement personnalisé",
            ]
        },
        'Clients Perdus': {
            'priorite': 'Win-back ou abandon',
            'couleur': 'red',
            'actions': [
                "Campagne win-back : offre exceptionnelle -20%",
                "Email 'Vous nous manquez' avec incentive fort",
            ]
        },
    }

    reco = base.get(rfm_segment, {
        'priorite': 'Segment inconnu',
        'couleur': 'gray',
        'actions': ["Recalculer le segment RFM avec des données complètes"]
    })

    # Affinage selon la probabilité de churn
    if rfm_segment == 'Clients Champions' and churn_proba >= 0.40:
        reco['actions'].insert(0, "⚠ Signal d'alerte : comportement récent dégradé")

    if rfm_segment == 'Clients Fideles' and churn_proba >= 0.50:
        reco['actions'].insert(0, "⚠ Risque élevé malgré la fidélité")

    return reco

# PRÉDICTION DEPUIS FORMULAIRE FLASK

def predict_churn_from_form(form: dict) -> dict:
    """
    Point d'entrée Flask.

    Flux :
        1. Calcul segment RFM (clustering — info marketing uniquement)
        2. Conversion formulaire → features (inclut RFMSegment feature modèle)
        3. align_features + imputation + prédiction RF
    """
    model, imputer, features = load_artifacts()

    # Étape 1 — RFM clustering (marketing uniquement, pas injecté dans le RF)
    rfm_cluster, rfm_segment = calculer_rfm_cluster(
        recency   = float(form.get('Recency',       50)),
        frequency = float(form.get('Frequency',      3)),
        monetary  = float(form.get('MonetaryTotal', 500)),
    )

    # Étape 2 — Features formulaire (inclut RFMSegment encodée)
    feat_dict = form_to_features(form)

    # Étape 3 — Prédiction
    df_input    = pd.DataFrame([feat_dict])
    X           = preprocess(df_input, imputer, features)
    churn_pred  = int(model.predict(X)[0])
    churn_proba = float(model.predict_proba(X)[0, 1])

    # Comptage des features réellement renseignées (non-NaN avant imputation)
    features_used = int(
        align_features(df_input.copy(), features).notna().sum(axis=1).iloc[0]
    )
    
    reg_result = predict_monetary_from_form(form)


    return {
        'churn_predicted'  : churn_pred,
        'churn_probability': round(churn_proba * 100, 1),
        'risk_segment'     : risk_label(churn_proba),
        'rfm_segment'      : rfm_segment,          # ← clustering (marketing)
        'rfm_cluster'      : rfm_cluster,
        'features_used'    : features_used,
        'features_total'   : len(features),
        'marketing_reco'   : marketing_recommendations(rfm_segment, churn_proba),
        'monetary_predicted': reg_result['monetary_predicted'],
        'monetary_model'    : reg_result['monetary_model'],
    }

# PRÉDICTION SUR CSV (CLI)

def predict_churn(df_input: pd.DataFrame) -> pd.DataFrame:
    """Prédit le churn sur un DataFrame CSV — retourne les colonnes essentielles."""
    results = predict_churn_with_rfm(df_input)
    return results[['CustomerID', 'churn_predicted', 'churn_probability', 'risk_segment']]


def predict_churn_with_rfm(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Prédit le churn + calcule le segment RFM (clustering) pour chaque ligne.
    RFMSegment est utilisée comme feature du modèle si présente dans le CSV.
    """
    model, imputer, features = load_artifacts()

    rfm_cols_present = {'Recency', 'Frequency', 'MonetaryTotal'}.issubset(df_input.columns)

    rows         = []
    rfm_segments = []

    for _, row in df_input.iterrows():
        feat_dict = row.to_dict()

        # Si RFMSegment est texte dans le CSV, on l'encode en ordinal
        if 'RFMSegment' in feat_dict:
            rfm_val = feat_dict['RFMSegment']
            if isinstance(rfm_val, str):
                feat_dict['RFMSegment'] = float(RFM_SEGMENT_MAP.get(rfm_val, 0))

        # Clustering RFM (marketing)
        if rfm_cols_present:
            _, rfm_segment = calculer_rfm_cluster(
                recency   = float(row.get('Recency',       50)),
                frequency = float(row.get('Frequency',      3)),
                monetary  = float(row.get('MonetaryTotal', 500)),
            )
        else:
            rfm_segment = 'N/A'

        rows.append(feat_dict)
        rfm_segments.append(rfm_segment)

    all_feats = pd.DataFrame(rows)

    # Comptage des features non-NaN AVANT imputation
    aligned               = align_features(all_feats.copy(), features)
    features_used_per_row = aligned.notna().sum(axis=1).tolist()

    X           = preprocess(all_feats, imputer, features)
    churn_pred  = model.predict(X)
    churn_proba = model.predict_proba(X)[:, 1]

    customer_ids = (df_input['CustomerID'].values
                    if 'CustomerID' in df_input.columns
                    else range(len(df_input)))

    return pd.DataFrame({
        'CustomerID'       : customer_ids,
        'churn_predicted'  : churn_pred,
        'churn_probability': churn_proba.round(4),
        'risk_segment'     : [risk_label(p) for p in churn_proba],
        'rfm_segment'      : rfm_segments,
        'features_used'    : features_used_per_row,
        'features_total'   : len(features),
    })


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
    import joblib
    cols = joblib.load('models/feature_columns.pkl')
    print([c for c in cols if c in ['MonetaryTotal','Recency','RFMSegment','CustomerTenureDays','FirstPurchaseDaysAgo']])
    args = parse_args()

    if args.client:
        data     = (json.load(open(args.client, encoding='utf-8'))
                    if args.client.endswith('.json')
                    else json.loads(args.client))
        df_input = pd.DataFrame([data])
        print(f"[INFO] Prédiction pour 1 client JSON")

    elif args.input:
        df_input = pd.read_csv(args.input)
        print(f"[INFO] Données chargées : {df_input.shape}")

    else:
        print("[ERREUR] Fournissez --input <csv> ou --client <json>")
        sys.exit(1)

    results = (predict_churn_with_rfm(df_input)
               if args.mode == 'all'
               else predict_churn(df_input))

    out_path = args.output or os.path.join(OUT_DIR, 'predictions_churn.csv')
    print("\n-- Résultats ----------------------------------------")
    print(results.to_string(index=False))
    results.to_csv(out_path, index=False)
    print(f"\n[OK] Prédictions sauvegardées : {out_path}")