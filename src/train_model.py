"""
train_model.py
==============
Pipeline complet d'entraînement — 3 modèles :
  1. Clustering    : K-Means RFM  (segmentation clients)
  2. Classification: Random Forest (prédiction Churn)
  3. Régression    : Comparaison XGBoost / LightGBM / RandomForest
                     → meilleur modèle sauvegardé automatiquement

Usage :
    python src/train_model.py
"""

import seaborn as sns
import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection  import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.cluster           import KMeans
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics           import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
    classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from imblearn.over_sampling    import SMOTE

warnings.filterwarnings('ignore')

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN    = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
DATA_TT    = os.path.join(BASE_DIR, 'data', 'train_test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR  = os.path.join(BASE_DIR, 'reports', 'figures')

for d in [DATA_TT, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constantes ────────────────────────────────────────────────────────────────
RFM_COLS = ['Recency', 'Frequency', 'MonetaryTotal']

ALL_SEGMENTS = ['Clients Champions', 'Clients Fideles', 'Clients Perdus', 'Clients a Risque']

SEP = "=" * 62

def splitter(df):
    """Split stratifié 80/20 — suppression leakage AVANT split (liste fixe)."""
    print("\n[A1] Split Train / Test (80/20 stratifié)...")

    y = df['Churn']
    X = df.drop(columns=['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
     # ── Extraire RFM AVANT de supprimer Recency ──
    X_train_rfm = X_train[RFM_COLS].copy()
    X_test_rfm  = X_test[RFM_COLS].copy()

    # ── Supprimer Recency du flux classification ──
    X_train = X_train.drop(columns=['Recency','TenureRatio','CustomerType','CustomerTenureDays','RFMSegment'],  errors='ignore')
    X_test  = X_test.drop(columns=['Recency','TenureRatio','CustomerType','CustomerTenureDays','RFMSegment'],  errors='ignore')
    print(f"    X_train : {X_train.shape} | X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test, X_train_rfm, X_test_rfm

def imputer_manquants(X_train, X_test):
    print("\n[A3] Imputation par médiane...")
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
    X_test  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns,  index=X_test.index)
    joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer.pkl'))
    return X_train, X_test

def target_encoding(X_train, X_test, y_train):
    print("\n[A2] Target Encoding — Country...")
    if 'Country' not in X_train.columns:
        print("    Country absent, étape ignorée.")
        return X_train, X_test

    temp          = X_train.copy()
    temp['Churn'] = y_train.values
    mapping       = temp.groupby('Country')['Churn'].mean()
    moy_globale   = y_train.mean()

    X_train['Country_encoded'] = X_train['Country'].map(mapping).fillna(moy_globale)
    X_test['Country_encoded']  = X_test['Country'].map(mapping).fillna(moy_globale)
    X_train = X_train.drop(columns=['Country'])
    X_test  = X_test.drop(columns=['Country'])

    joblib.dump({'mapping': mapping, 'moy_globale': moy_globale},
                os.path.join(MODELS_DIR, 'country_encoding.pkl'))
    print(f"    Country → Country_encoded | X_train : {X_train.shape}")
    return X_train, X_test

def standardiser_features(X_train, X_test):
    print("    Standardisation avec StandardScaler...")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"    Train scaled : {X_train_scaled.shape} | Test scaled : {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler

def appliquer_smote(X_train, y_train):
    print("\n[A5] SMOTE — Rééquilibrage...")
    print(f"    Avant : {y_train.value_counts().to_dict()}")
    smote        = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"    Après : {pd.Series(y_bal).value_counts().to_dict()} | Shape : {X_bal.shape}")
    return X_bal, y_bal

# ══════════════════════════════════════════════════════════════
# PARTIE A — CLUSTERING RFM
# ══════════════════════════════════════════════════════════════

def entrainer_clustering(X_train, X_test, y_train):
    """
    Clustering RFM complet sur df entier (segmentation client).
    Différent de la classification : pas de contrainte de leakage sur Recency.
    """
    print("\n" + "="*60)
    print("  VOIE 1 — CLUSTERING RFM")
    print("="*60)

    # ── ÉTAPE 1 : Extraction des colonnes RFM ──
    rfm_train = X_train[RFM_COLS].copy()
    rfm_test  = X_test[RFM_COLS].copy()
    
    # ── ÉTAPE 2 : Imputation des NaN (médiane) ──
    print("\n[RFM-1] Imputation des valeurs manquantes dans RFM...")
    imputer_rfm = SimpleImputer(strategy='median')
    rfm_train = pd.DataFrame(
        imputer_rfm.fit_transform(rfm_train),
        columns=RFM_COLS,
        index=X_train.index
    )
    rfm_test = pd.DataFrame(
        imputer_rfm.transform(rfm_test),
        columns=RFM_COLS,
        index=X_test.index
    )
    print(f"    RFM Train : {rfm_train.shape} | NaN : {rfm_train.isna().sum().sum()}")
    print(f"    RFM Test  : {rfm_test.shape}  | NaN : {rfm_test.isna().sum().sum()}")
    
    # ── ÉTAPE 3 : Transformation log ──
    print("\n[RFM-2] Transformation logarithmique...")
    rfm_train = rfm_train.clip(lower=0).apply(np.log1p)
    rfm_test  = rfm_test.clip(lower=0).apply(np.log1p)
    
    # ── ÉTAPE 4 : Standardisation ──
    print("\n[RFM-3] Standardisation...")
    scaler = StandardScaler()
    rfm_train_sc = scaler.fit_transform(rfm_train)
    rfm_test_sc  = scaler.transform(rfm_test)
    
    # ── K-Means (fit sur train) ──
    print("\n[RFM-4] Entraînement K-Means (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rfm_train_sc)
    
    # ── Prédiction sur les deux sets (mêmes centroïdes) ──
    clusters_train = pd.Series(
        kmeans.predict(rfm_train_sc),
        index=X_train.index, name='Cluster_RFM'
    )
    clusters_test = pd.Series(
        kmeans.predict(rfm_test_sc),
        index=X_test.index, name='Cluster_RFM'
    )
    
    print(f"\n[RFM-5] Résultats du clustering :")
    print(f"    Clusters TRAIN : {clusters_train.value_counts().sort_index().to_dict()}")
    print(f"    Clusters TEST  : {clusters_test.value_counts().sort_index().to_dict()}")
    
    # ── Profiling & nommage (sur train uniquement) ──
    print("\n[RFM-6] Profilage des segments...")
    df_profile = rfm_train.copy()
    df_profile['Churn'] = y_train.values
    df_profile['Cluster'] = clusters_train.values
    
    # Retransformation inverse pour affichage (enlever log)
    for col in RFM_COLS:
        df_profile[col] = np.expm1(df_profile[col])
    
    profiles = df_profile.groupby('Cluster')[RFM_COLS + ['Churn']].mean().round(2)
    profiles['nb_clients'] = df_profile['Cluster'].value_counts().sort_index()
    profiles['churn_%'] = (profiles['Churn'] * 100).round(1)
    
    def nommer(row):
        if row['churn_%'] >= 60:   return 'Clients Perdus'
        if row['churn_%'] >= 15:   return 'Clients a Risque'
        if row['Frequency'] >= profiles['Frequency'].median(): return 'Clients Champions'
        return 'Clients Fideles'
    
    profiles['segment'] = profiles.apply(nommer, axis=1)
    segment_map = profiles['segment'].to_dict()
    
    print("\n    Profils des clusters (basés sur TRAIN) :")
    print(profiles[['Recency', 'Frequency', 'MonetaryTotal', 'churn_%', 'nb_clients', 'segment']].to_string())
    
    # ── Sauvegarde ──
    joblib.dump(kmeans,      os.path.join(MODELS_DIR, 'kmeans_rfm.pkl'))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, 'scaler_rfm.pkl'))
    joblib.dump(imputer_rfm, os.path.join(MODELS_DIR, 'imputer_rfm.pkl'))
    joblib.dump(segment_map, os.path.join(MODELS_DIR, 'cluster_segment_map.pkl'))
    print("\n    Artefacts RFM sauvegardés (imputer_rfm, scaler_rfm, kmeans_rfm, cluster_segment_map).")
    
    return clusters_train, clusters_test, segment_map

# PARTIE C — CLASSIFICATION

def entrainer_classification(X_train_bal, y_train_bal, X_test, y_test):
    print(f"\n{SEP}")
    print("  PARTIE C — CLASSIFICATION (Churn)")
    print(SEP)
    print("\n[C1] GridSearchCV — Random Forest Classifier")

    param_grid = {
        'n_estimators'    : [200, 300, 500],
        'max_depth'       : [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_bal, y_train_bal)

    best_rf = grid.best_estimator_
    print(f"\n    Meilleurs params : {grid.best_params_}")
    print(f"    AUC-ROC (CV)     : {grid.best_score_*100:.1f}%")

    y_pred  = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    print("\n[C2] Évaluation — Test Set")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Fidele (0)', 'Parti (1)']))
    print(f"    Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"    F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"    AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")

    importances = pd.Series(best_rf.feature_importances_,
                            index=X_test.columns).sort_values(ascending=False)
    print(f"\n    Top 10 features :")
    print(importances.head(10).to_string())

    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
    print("\n    random_forest_churn.pkl sauvegardé")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predit Fidele', 'Predit Churn'],
                yticklabels=['Reel Fidele',   'Reel Churn'])
    plt.title(f'Matrice de Confusion — Random Forest\n'
              f'Acc={accuracy_score(y_test,y_pred):.3f}  '
              f'AUC={roc_auc_score(y_test,y_proba):.3f}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clf_confusion_matrix.png'), dpi=120)
    plt.close()
    
    return best_rf, y_pred, y_proba

# PARTIE D — RÉGRESSION AVEC COMPARAISON DE MODÈLES

def _mape(y_true, y_pred):
    """MAPE filtré sur valeurs >= 10 pour éviter les divisions par zéro."""
    mask = np.array(y_true) >= 10
    if mask.sum() == 0:
        return np.nan
    yt = np.array(y_true)[mask]
    yp = np.array(y_pred)[mask]
    return np.mean(np.abs((yt - yp) / yt)) * 100


def _rmsle(y_true, y_pred):
    """Root Mean Squared Log Error."""
    y_pred_clipped = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))


def _evaluer(nom, y_true, y_pred):
    """Calcule et affiche toutes les métriques pour un modèle."""
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = _mape(y_true, y_pred)
    rmsle = _rmsle(y_true, y_pred)
    print(f"\n  [{nom}]")
    print(f"    MAE   : {mae:.2f} €")
    print(f"    RMSE  : {rmse:.2f} €")
    print(f"    R2    : {r2:.4f}")
    print(f"    MAPE  : {mape:.1f}%   ← métrique principale")
    print(f"    RMSLE : {rmsle:.4f}")
    return {'nom': nom, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'rmsle': rmsle}


def entrainer_regression(df, X_train, X_test):
    print(f"\n{SEP}")
    print("  PARTIE D — RÉGRESSION MonetaryTotal (comparaison de modèles)")
    print(SEP)

    TARGET = 'MonetaryTotal'
    if TARGET not in df.columns:
        print(f"    {TARGET} absent — régression ignorée.")
        return None

    # ── Préparation ────────────────────────────────────────────
    print("\n[D1] Préparation des données...")

    y_train_raw = df.loc[X_train.index, TARGET]
    y_test_raw  = df.loc[X_test.index,  TARGET]

    mask_train = y_train_raw.notna()
    mask_test  = y_test_raw.notna()

    y_train_reg = y_train_raw[mask_train]
    y_test_reg  = y_test_raw[mask_test]
    X_train_reg = X_train[mask_train].copy()
    X_test_reg  = X_test[mask_test].copy()

    print(f"    Train : {X_train_reg.shape} | Test : {X_test_reg.shape}")
    print(f"    Moy MonetaryTotal : {y_train_reg.mean():.1f} € | Std : {y_train_reg.std():.1f} €")

    # ── Transformation log ─────────────────────────────────────
    print("\n[D1.5] Transformation logarithmique de la cible...")
    y_train_log = np.log1p(y_train_reg)
    y_test_log  = np.log1p(y_test_reg)
    print(f"    Avant log : μ={y_train_reg.mean():.1f}  σ={y_train_reg.std():.1f}")
    print(f"    Après log : μ={y_train_log.mean():.2f}  σ={y_train_log.std():.2f}")

    # ── Définition des modèles candidats ───────────────────────
    print("\n[D2] Comparaison de modèles (cross-validation 5 folds sur log-target)...")

    candidats = {
        'XGBoost': {
            'modele': XGBRegressor(random_state=42, objective='reg:squarederror',
                                   verbosity=0),
            'grille': {
                'n_estimators'    : [100, 200],
                'max_depth'       : [5, 7, 10],
                'learning_rate'   : [0.05, 0.1, 0.2],
                'min_child_weight': [1, 3],
            }
        },
        'LightGBM': {
            'modele': LGBMRegressor(random_state=42, verbosity=-1),
            'grille': {
                'n_estimators'  : [100, 200],
                'max_depth'     : [5, 7, 10],
                'learning_rate' : [0.05, 0.1, 0.2],
                'num_leaves'    : [31, 63],
            }
        },
        'RandomForest': {
            'modele': RandomForestRegressor(random_state=42, n_jobs=-1),
            'grille': {
                'n_estimators': [100, 200],
                'max_depth'   : [10, 20, None],
                'max_features': ['sqrt', 'log2'],
            }
        },
    }

    resultats_cv  = {}
    meilleurs     = {}

    for nom, config in candidats.items():
        print(f"\n  --- {nom} ---")
        grid = GridSearchCV(
            config['modele'], config['grille'],
            cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_reg, y_train_log)
        meilleurs[nom]    = grid.best_estimator_
        resultats_cv[nom] = grid.best_score_
        print(f"    Meilleurs params : {grid.best_params_}")
        print(f"    R2 CV (log)      : {grid.best_score_:.4f}")

    # ── Évaluation sur le test set (échelle originale) ─────────
    print(f"\n[D3] Évaluation sur le Test Set (échelle originale) :")
    print("=" * 55)

    scores_test = {}
    for nom, modele in meilleurs.items():
        y_pred_log = modele.predict(X_test_reg)
        y_pred     = np.expm1(y_pred_log)
        scores_test[nom] = _evaluer(nom, y_test_reg, y_pred)

    # ── Sélection du meilleur modèle ───────────────────────────
    print(f"\n[D4] Sélection du meilleur modèle...")
    print("-" * 55)

    # Tableau récapitulatif
    print(f"\n  {'Modèle':<15} {'MAPE%':<10} {'RMSLE':<10} {'R2':<8} {'MAE €':<10}")
    print(f"  {'-'*53}")
    for nom, s in scores_test.items():
        print(f"  {nom:<15} {s['mape']:<10.1f} {s['rmsle']:<10.4f} "
              f"{s['r2']:<8.4f} {s['mae']:<10.2f}")

    meilleur_nom = min(scores_test, key=lambda n: scores_test[n]['mape'])
    meilleur_modele = meilleurs[meilleur_nom]
    print(f"\n  Meilleur modèle : {meilleur_nom} "
          f"(MAPE={scores_test[meilleur_nom]['mape']:.1f}%)")

    # ── Sauvegarde ─────────────────────────────────────────────
    joblib.dump(meilleur_modele, os.path.join(MODELS_DIR, 'regressor_monetary.pkl'))
    joblib.dump({
        'use_log_transform' : True,
        'model_type'        : meilleur_nom.lower(),
        'feature_columns'   : X_train_reg.columns.tolist(),
        'mape_test'         : scores_test[meilleur_nom]['mape'],
        'rmsle_test'        : scores_test[meilleur_nom]['rmsle'],
        'r2_test'           : scores_test[meilleur_nom]['r2'],
    }, os.path.join(MODELS_DIR, 'regression_metadata.pkl'))

    print(f"\n    regressor_monetary.pkl sauvegardé ({meilleur_nom})")
    print(f"    regression_metadata.pkl sauvegardé")

    # ── Graphique comparaison ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    noms   = list(scores_test.keys())
    colors = ['#378ADD', '#1D9E75', '#E24B4A']

    for ax, metrique, label in zip(
        axes,
        ['mape', 'rmsle', 'r2'],
        ['MAPE % (↓ meilleur)', 'RMSLE (↓ meilleur)', 'R² (↑ meilleur)']
    ):
        valeurs = [scores_test[n][metrique] for n in noms]
        bars = ax.bar(noms, valeurs, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(metrique.upper())
        for bar, val in zip(bars, valeurs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(valeurs)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        # Mettre en évidence le meilleur
        idx_best = noms.index(meilleur_nom)
        bars[idx_best].set_edgecolor('black')
        bars[idx_best].set_linewidth(2)

    plt.suptitle('Comparaison des modèles de régression — MonetaryTotal',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    chemin_plot = os.path.join(PLOTS_DIR, 'regression_model_comparison.png')
    plt.savefig(chemin_plot, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Graphique : {chemin_plot}")

    return meilleur_modele


# PIPELINE PRINCIPAL

def main():
    print(SEP)
    print("   TRAIN_MODEL.PY — Clustering + Classification + Régression")
    print(SEP)

    print("\n[1] Chargement des données...")
    df = pd.read_csv(DATA_IN)
    print(f"    Shape : {df.shape}")
    print(f"    Churn : {df['Churn'].value_counts().to_dict()}")

    # Nettoyage
    if 'MonetaryTotal' in df.columns:
        df.loc[df['MonetaryTotal'] < 0, 'MonetaryTotal'] = np.nan
    
    # ── ÉTAPE 1 : Split ──
    X_train, X_test, y_train, y_test, X_train_rfm, X_test_rfm = splitter(df)

    # ── ÉTAPE 2 : Clustering RFM ──
    clusters_train, clusters_test, segment_map = entrainer_clustering(
        X_train_rfm, X_test_rfm, y_train
    )
    # ── ÉTAPE 3 : Preprocessing Classification ──
    print(f"\n{SEP}")
    print("  PARTIE B — PREPROCESSING CLASSIFICATION")
    print(SEP)

    X_train, X_test = target_encoding(X_train, X_test, y_train)
    X_train, X_test = imputer_manquants(X_train, X_test)

    # ── STANDARDISATION ──
    print("\n[A4] Standardisation des features...")
    X_train, X_test, scaler = standardiser_features(X_train, X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print(f"    scaler.pkl sauvegardé")

    # Sauvegarde de la liste exacte des features
    joblib.dump(X_train.columns.tolist(),
                os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    print(f"\n    feature_columns.pkl sauvegardé ({len(X_train.columns)} features)")

    # ── ÉTAPE 4 : SMOTE + Classification ──
    X_train_bal, y_train_bal = appliquer_smote(X_train, y_train)
    best_rf, _, _ = entrainer_classification(X_train_bal, y_train_bal, X_test, y_test)

    # ── ÉTAPE 5 : Régression ──
    entrainer_regression(df, X_train, X_test)

    # Sauvegarde CSV
    print(f"\n{SEP}")
    print("  SAUVEGARDE FINALE")
    print(SEP)
    X_train.to_csv(os.path.join(DATA_TT, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_TT, 'X_test.csv'),  index=False)
    y_train.to_csv(os.path.join(DATA_TT, 'y_train.csv'), index=False)
    y_test.to_csv( os.path.join(DATA_TT, 'y_test.csv'),  index=False)
    print("    CSV sauvegardés.")

    print(f"\n{SEP}")
    print("  Pipeline terminé avec succès !")
    print("  Artefacts models/ :")
    print("    RFM            : imputer_rfm, scaler_rfm, kmeans_rfm, cluster_segment_map")
    print("    Classification : imputer, scaler, random_forest_churn, feature_columns, country_encoding")
    print("    Régression     : regressor_monetary, regression_metadata")
    print(SEP)

if __name__ == '__main__':
    
    main()