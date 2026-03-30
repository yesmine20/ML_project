"""
train_model.py
==============
Pipeline complet d'entraînement — 3 modèles :
  1. Clustering   : K-Means RFM (segmentation clients)
  2. Classification: Random Forest (prédiction Churn)
  3. Régression   : Random Forest (prédiction MonetaryTotal)

Preprocessing post-split : leakage → split → target encoding →
imputation → outliers → normalisation → SMOTE → GridSearchCV.

Usage :
    python src/train_model.py
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')           # pas d'affichage interactif en script
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection   import train_test_split, GridSearchCV
from sklearn.preprocessing     import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.decomposition     import PCA
from sklearn.cluster           import KMeans
from sklearn.metrics           import silhouette_score
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics           import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
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
# Features leakage pour la classification (proxies de Recency / Churn)
# Recency reste dans df pour le clustering mais est exclue de la classification
LEAKAGE_CLF = [
    'ChurnRiskCategory', 'CustomerType', 'AccountStatus',
    'Recency',                 # frontière parfaite ≤90=fidèle / ≥91=parti
    'TenureRatio',             # Recency / CustomerTenureDays
    'MonetaryPerDay',          # MonetaryTotal / (Recency+1)
    'FirstPurchaseDaysAgo',    # tenure depuis 1er achat
    'CustomerTenureDays',      # durée de vie client
    'LoyaltyLevel',            # construit sur CustomerTenureDays
]

FEATURES_RFM = ['Recency', 'Frequency', 'MonetaryTotal']

SEGMENT_MAP = {
    0: 'Clients Champions',
    1: 'Clients Fidèles',
    2: 'Clients Perdus',
    3: 'Clients à Risque',
}

SEP = "=" * 62


# ══════════════════════════════════════════════════════════════
# PARTIE A — PREPROCESSING COMMUN
# ══════════════════════════════════════════════════════════════

def charger_donnees():
    """Étape 1 — Chargement de data_clean.csv"""
    print("\n[1] Chargement des données...")
    df = pd.read_csv(DATA_IN)
    print(f"    Shape : {df.shape}")
    manquants = df.isnull().sum()
    manquants = manquants[manquants > 0]
    if len(manquants):
        print("    NaN par colonne (imputés après split) :")
        print(manquants.to_string())
    return df


def splitter(df):
    """
    Étape 2 — Split stratifié 80/20 sur Churn.
    Supprime LEAKAGE_CLF APRÈS le split pour ne pas perdre Recency
    avant le clustering RFM (qui utilise df brut).
    """
    print("\n[2] Split Train / Test (80/20 stratifié)...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    supprimees = [c for c in LEAKAGE_CLF if c in X_train.columns]
    X_train = X_train.drop(columns=supprimees)
    X_test  = X_test.drop(columns=supprimees)

    print(f"    X_train : {X_train.shape} | X_test : {X_test.shape}")
    print(f"    Churn train : {y_train.mean()*100:.1f}%  |  "
          f"Churn test  : {y_test.mean()*100:.1f}%")
    print(f"    Leakage supprimé : {supprimees}")
    return X_train, X_test, y_train, y_test


def target_encoding(X_train, X_test, y_train):
    """Étape 3 — Target Encoding Country (fit sur train seulement)."""
    print("\n[3] Target Encoding — Country...")
    if 'Country' not in X_train.columns:
        print("    Country absent, étape ignorée.")
        return X_train, X_test

    temp = X_train.copy()
    temp['Churn'] = y_train.values
    mapping     = temp.groupby('Country')['Churn'].mean()
    moy_globale = y_train.mean()

    X_train['Country_encoded'] = X_train['Country'].map(mapping).fillna(moy_globale)
    X_test['Country_encoded']  = X_test['Country'].map(mapping).fillna(moy_globale)
    X_train = X_train.drop(columns=['Country'])
    X_test  = X_test.drop(columns=['Country'])

    joblib.dump({'mapping': mapping, 'moy_globale': moy_globale},
                os.path.join(MODELS_DIR, 'country_encoding.pkl'))
    print(f"    Country → Country_encoded  |  X_train : {X_train.shape}")
    return X_train, X_test


def imputer_manquants(X_train, X_test):
    """Étape 4 — Imputation médiane (fit sur train seulement)."""
    print("\n[4] Imputation par médiane...")
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
    X_test  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns,  index=X_test.index)
    joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer.pkl'))
    print(f"    NaN train : {X_train.isnull().sum().sum()}  |  "
          f"NaN test : {X_test.isnull().sum().sum()}")
    return X_train, X_test, imputer


def supprimer_outliers(X_train, y_train):
    """Étape 5 — Suppression outliers extrêmes Z-score > 4 (train seulement)."""
    print("\n[5] Suppression des outliers (Z-score > 4)...")
    z    = np.nan_to_num(np.abs(stats.zscore(X_train)), nan=0.0)
    mask = (z < 4).all(axis=1)
    n_out = (~mask).sum()
    X_train = X_train[mask]
    y_train = y_train[mask]
    print(f"    Outliers supprimés : {n_out}  |  X_train : {X_train.shape}")
    print(f"    Distribution Churn : {y_train.value_counts().to_dict()}")
    return X_train, y_train


def normaliser(X_train, X_test):
    """Étape 6 — StandardScaler (fit sur train seulement)."""
    print("\n[6] Normalisation StandardScaler...")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,  index=X_test.index)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print(f"    Moyenne ~ {X_train.mean().mean():.4f}  |  "
          f"Écart-type ~ {X_train.std().mean():.4f}")
    return X_train, X_test, scaler


def appliquer_smote(X_train, y_train):
    """Étape 7 — SMOTE sur X_train (après split, jamais sur X_test)."""
    print("\n[7] SMOTE — Rééquilibrage des classes...")
    print(f"    Avant : {y_train.value_counts().to_dict()}")
    smote        = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"    Après : {pd.Series(y_bal).value_counts().to_dict()}  |  "
          f"Shape : {X_bal.shape}")
    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════
# PARTIE B — CLUSTERING RFM (K-Means)
# ══════════════════════════════════════════════════════════════

def entrainer_clustering(df):
    """
    Étape 8 — Clustering K-Means sur les features RFM.

    Pipeline :
        données brutes (df)
        → dropna sur RFM
        → clip(lower=0)     : évite log(valeur_négative) = NaN
        → log1p             : compresse les distributions skewed
        → StandardScaler    : même échelle pour K-Means
        → K-Means k=4

    Retourne : kmeans_rfm, scaler_rfm, df_clust (avec colonnes cluster + segment)
    """
    print(f"\n{SEP}")
    print("  PARTIE B — CLUSTERING RFM")
    print(SEP)
    print("\n[8] Entraînement K-Means RFM...")

    # 8.1 Préparation
    df_rfm = df[FEATURES_RFM + ['Churn']].dropna(subset=FEATURES_RFM).reset_index(drop=True)
    print(f"    Clients disponibles : {len(df_rfm)}")

    # Vérification valeurs négatives
    for col in FEATURES_RFM:
        n_neg = (df_rfm[col] < 0).sum()
        if n_neg > 0:
            print(f"    ⚠️  {col} : {n_neg} valeurs négatives → clip(0)")

    # 8.2 Transformation : clip → log1p → StandardScaler
    df_rfm_log  = df_rfm[FEATURES_RFM].clip(lower=0).apply(np.log1p)
    scaler_rfm  = StandardScaler()
    X_rfm_sc    = scaler_rfm.fit_transform(df_rfm_log)
    assert not np.isnan(X_rfm_sc).any(), "NaN dans X_rfm_sc !"

    # 8.3 Choix du K : coude + silhouette
    print("\n    Calcul coude + silhouette (K=2..8)...")
    inertias, silhouettes = [], []
    K_range = range(2, 9)
    for k in K_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_rfm_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_rfm_sc, lbl,
                                            sample_size=2000, random_state=42))

    k_optimal = 4
    print(f"    K retenu : {k_optimal}  |  "
          f"Silhouette : {silhouettes[k_optimal-2]:.3f}")

    # Sauvegarde graphique coude + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(K_range, inertias, 'bo-', markersize=7)
    axes[0].axvline(k_optimal, color='r', linestyle='--', label=f'K={k_optimal}')
    axes[0].set(title='Méthode du Coude', xlabel='K', ylabel='Inertie')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(K_range, silhouettes, 'go-', markersize=7)
    axes[1].axvline(k_optimal, color='r', linestyle='--', label=f'K={k_optimal}')
    axes[1].set(title='Score de Silhouette', xlabel='K', ylabel='Silhouette')
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle('Choix du K — Clustering RFM', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'rfm_elbow_silhouette.png'), dpi=120)
    plt.close()

    # 8.4 K-Means k=4
    kmeans_rfm = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans_rfm.fit(X_rfm_sc)

    df_clust = df_rfm.copy()
    df_clust.columns = FEATURES_RFM + ['churn']
    df_clust['cluster'] = kmeans_rfm.labels_

    # 8.5 Profils & nommage automatique
    profiles = df_clust.groupby('cluster')[FEATURES_RFM + ['churn']].mean().round(2)
    profiles['nb_clients'] = df_clust['cluster'].value_counts().sort_index()
    profiles['pct_%']      = (profiles['nb_clients'] / len(df_clust) * 100).round(1)
    profiles['churn_%']    = (profiles['churn'] * 100).round(1)

    def nommer(row):
        if row['churn_%'] >= 60: return 'Clients Perdus'
        if row['churn_%'] >= 15: return 'Clients à Risque'
        if row['Frequency'] >= profiles['Frequency'].median(): return 'Clients Champions'
        return 'Clients Fidèles'

    profiles['segment'] = profiles.apply(nommer, axis=1)
    df_clust['segment'] = df_clust['cluster'].map(profiles['segment'].to_dict())

    print("\n    Profils des clusters :")
    print(profiles[['Recency', 'Frequency', 'MonetaryTotal',
                     'churn_%', 'nb_clients', 'pct_%', 'segment']].to_string())

    # 8.6 Visualisation ACP 2D
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d   = pca_2d.fit_transform(X_rfm_sc)
    colors = ['#E24B4A', '#378ADD', '#639922', '#BA7517']

    plt.figure(figsize=(10, 7))
    for i in range(k_optimal):
        mask = df_clust['cluster'] == i
        seg  = profiles.loc[i, 'segment']
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[i], label=f'C{i} — {seg} (n={mask.sum()})',
                    alpha=0.55, s=20)
    plt.title('Segmentation RFM — K-Means k=4 (ACP 2D)', fontsize=13)
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'rfm_clusters_pca2d.png'), dpi=120)
    plt.close()

    # 8.7 Sauvegarde artefacts RFM
    joblib.dump(kmeans_rfm, os.path.join(MODELS_DIR, 'kmeans_rfm.pkl'))
    joblib.dump(scaler_rfm, os.path.join(MODELS_DIR, 'scaler_rfm.pkl'))
    print("\n    kmeans_rfm.pkl + scaler_rfm.pkl sauvegardés")

    return kmeans_rfm, scaler_rfm, df_clust


# ══════════════════════════════════════════════════════════════
# PARTIE C — CLASSIFICATION (Prédiction Churn)
# ══════════════════════════════════════════════════════════════

def entrainer_classification(X_train_bal, y_train_bal, X_test, y_test):
    """
    Étape 9 — Random Forest Classifier avec GridSearchCV.
    Évaluation : classification_report + matrice de confusion + courbe ROC.
    """
    print(f"\n{SEP}")
    print("  PARTIE C — CLASSIFICATION (Churn)")
    print(SEP)
    print("\n[9] GridSearchCV — Random Forest Classifier...")
    print("    (~2-3 minutes)\n")

    param_grid = {
        'n_estimators'     : [100, 200],
        'max_depth'        : [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf' : [2, 4],
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_bal, y_train_bal)

    best_rf = grid.best_estimator_
    print(f"\n    Meilleurs params  : {grid.best_params_}")
    print(f"    AUC-ROC (CV)     : {grid.best_score_*100:.1f}%")

    # Prédictions
    y_pred  = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]  # ← probabilités pour AUC

    # ── Métriques ────────────────────────────────────────────
    print("\n[10] Évaluation Classification — Test Set")
    print("-" * 50)
    print(classification_report(y_test, y_pred,
                                target_names=['Fidèle (0)', 'Parti (1)']))
    print(f"    Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"    F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"    AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")   # ← proba !

    # ── Matrice de confusion ─────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    Matrice de confusion :")
    print(f"    {'':15s} Prédit Fidèle  Prédit Churn")
    print(f"    {'Réel Fidèle':15s}  {cm[0,0]:^13d}  {cm[0,1]:^12d}")
    print(f"    {'Réel Churn':15s}  {cm[1,0]:^13d}  {cm[1,1]:^12d}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Prédit Fidèle', 'Prédit Churn'],
                yticklabels=['Réel Fidèle',   'Réel Churn'])
    plt.title(f'Matrice de Confusion — Random Forest\n'
              f'Accuracy={accuracy_score(y_test,y_pred):.3f}  '
              f'AUC={roc_auc_score(y_test,y_proba):.3f}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clf_confusion_matrix.png'), dpi=120)
    plt.close()

    # ── Feature importance ───────────────────────────────────
    importances = pd.Series(best_rf.feature_importances_,
                            index=X_test.columns).sort_values(ascending=False)
    top20 = importances.head(20)
    print(f"\n    Top 10 features :")
    print(top20.head(10).to_string())

    plt.figure(figsize=(10, 7))
    sns.barplot(x=top20.values, y=top20.index, hue=top20.index,
                palette='Blues_r', legend=False)
    plt.title('Top 20 Features Importance — Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clf_feature_importance.png'), dpi=120)
    plt.close()

    # ── Sauvegarde ───────────────────────────────────────────
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
    print("\n    random_forest_churn.pkl sauvegardé")

    return best_rf, y_pred, y_proba


# ══════════════════════════════════════════════════════════════
# PARTIE D — RÉGRESSION (Prédiction MonetaryTotal)
# ══════════════════════════════════════════════════════════════

def entrainer_regression(df, X_train, X_test, y_train):
    """
    Étape 11 — Random Forest Regressor avec GridSearchCV.
    Cible : MonetaryTotal (montant total dépensé par client).
    Évaluation : MAE, RMSE, R².

    Note : on reconstruit y_reg depuis df pour que les index
    correspondent aux lignes conservées après outliers/SMOTE.
    """
    print(f"\n{SEP}")
    print("  PARTIE D — RÉGRESSION (MonetaryTotal)")
    print(SEP)

    TARGET_REG = 'MonetaryTotal'
    if TARGET_REG not in df.columns:
        print(f"\n    ⚠️  {TARGET_REG} absent du dataset — régression ignorée.")
        return None

    print(f"\n[11] Préparation des données de régression...")

    # Récupérer MonetaryTotal depuis les index de X_train et X_test
    y_train_reg = df.loc[X_train.index, TARGET_REG] if TARGET_REG in df.columns else None
    y_test_reg  = df.loc[X_test.index,  TARGET_REG] if TARGET_REG in df.columns else None

    # Si MonetaryTotal a été supprimé du X (il est dans df mais pas dans X_train)
    # on le récupère depuis data_clean directement
    if y_train_reg is None or y_train_reg.isnull().all():
        print("    MonetaryTotal non trouvé dans X_train.index → chargement depuis df...")
        return None

    print(f"    y_train_reg : {y_train_reg.shape}  |  "
          f"y_test_reg : {y_test_reg.shape}")
    print(f"    Moyenne MonetaryTotal : {y_train_reg.mean():.1f} £  |  "
          f"Std : {y_train_reg.std():.1f} £")

    print("\n[12] GridSearchCV — Random Forest Regressor...")
    print("    (~1-2 minutes)\n")

    param_grid_reg = {
        'n_estimators': [100, 200],
        'max_depth'   : [5, 10, 20],
        'min_samples_split': [2, 5],
    }

    grid_reg = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_reg, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    grid_reg.fit(X_train, y_train_reg)

    best_reg = grid_reg.best_estimator_
    print(f"\n    Meilleurs params : {grid_reg.best_params_}")
    print(f"    R² (CV)          : {grid_reg.best_score_:.4f}")

    # ── Évaluation ───────────────────────────────────────────
    y_pred_reg = best_reg.predict(X_test)

    mae  = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2   = r2_score(y_test_reg, y_pred_reg)

    print("\n[13] Évaluation Régression — Test Set")
    print("-" * 40)
    print(f"    MAE  (erreur absolue moyenne) : {mae:.2f} £")
    print(f"    RMSE (erreur quadratique)     : {rmse:.2f} £")
    print(f"    R²   (variance expliquée)     : {r2:.4f}")

    # ── Graphique Réel vs Prédit ─────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.3, s=15, color='#378ADD')
    lim = [min(y_test_reg.min(), y_pred_reg.min()),
           max(y_test_reg.max(), y_pred_reg.max())]
    plt.plot(lim, lim, 'r--', lw=1.5, label='Prédiction parfaite')
    plt.xlabel('MonetaryTotal Réel (£)')
    plt.ylabel('MonetaryTotal Prédit (£)')
    plt.title(f'Régression — Réel vs Prédit\nMAE={mae:.1f} £  RMSE={rmse:.1f} £  R²={r2:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'reg_pred_vs_real.png'), dpi=120)
    plt.close()

    # ── Sauvegarde ───────────────────────────────────────────
    joblib.dump(best_reg, os.path.join(MODELS_DIR, 'rf_regressor_monetary.pkl'))
    print("\n    rf_regressor_monetary.pkl sauvegardé")

    return best_reg


# ══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("   TRAIN_MODEL.PY — Pipeline Clustering + Classification + Régression")
    print(SEP)

    # ── 1. Chargement ────────────────────────────────────────
    df = charger_donnees()

    # ── PARTIE B : Clustering RFM (sur df brut, AVANT split) ─
    # On passe df entier car Recency est nécessaire pour le clustering
    # et sera supprimée du pipeline classification ensuite
    kmeans_rfm, scaler_rfm, df_clust = entrainer_clustering(df)

    # ── PARTIE A suite : Preprocessing classification ────────
    print(f"\n{SEP}")
    print("  PARTIE A — PRÉTRAITEMENT CLASSIFICATION")
    print(SEP)

    X_train, X_test, y_train, y_test = splitter(df)
    X_train, X_test                  = target_encoding(X_train, X_test, y_train)
    X_train, X_test, imputer         = imputer_manquants(X_train, X_test)
    X_train, y_train                 = supprimer_outliers(X_train, y_train)
    X_train, X_test, scaler          = normaliser(X_train, X_test)

    # Sauvegarde liste de features (pour predict.py)
    joblib.dump(X_train.columns.tolist(),
                os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    print("\n    feature_columns.pkl sauvegardé")

    X_train_bal, y_train_bal = appliquer_smote(X_train, y_train)

    # ── PARTIE C : Classification ────────────────────────────
    best_rf, y_pred_clf, y_proba_clf = entrainer_classification(
        X_train_bal, y_train_bal, X_test, y_test
    )

    # ── PARTIE D : Régression ────────────────────────────────
    best_reg = entrainer_regression(df, X_train, X_test, y_train)

    # ── Sauvegarde train/test ────────────────────────────────
    print(f"\n{SEP}")
    print("  SAUVEGARDE FINALE")
    print(SEP)
    X_train.to_csv(os.path.join(DATA_TT, 'X_train.csv'), index=False)
    X_test.to_csv( os.path.join(DATA_TT, 'X_test.csv'),  index=False)
    y_train.to_csv(os.path.join(DATA_TT, 'y_train.csv'), index=False)
    y_test.to_csv( os.path.join(DATA_TT, 'y_test.csv'),  index=False)
    df_clust.to_csv(os.path.join(DATA_TT, 'df_clust_rfm.csv'), index=False)
    print("    X_train / X_test / y_train / y_test / df_clust_rfm sauvegardés")

    print(f"\n{SEP}")
    print("   ✅  Pipeline terminé avec succès !")
    print(f"   Artefacts dans models/")
    print("     Classification : random_forest_churn.pkl, imputer.pkl,")
    print("                      scaler.pkl, feature_columns.pkl, country_encoding.pkl")
    print("     Clustering     : kmeans_rfm.pkl, scaler_rfm.pkl")
    if best_reg:
        print("     Régression     : rf_regressor_monetary.pkl")
    print(f"   Graphiques dans reports/figures/")
    print(SEP)


if __name__ == '__main__':
    main()