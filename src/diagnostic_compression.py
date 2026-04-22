"""
diagnostic_compression.py
==========================
Script de diagnostic pour comprendre POURQUOI les probabilités
du Random Forest sont compressées entre 30% et 55%.

Ce script ne modifie rien. Il pose 3 questions :
  Q1 — Est-ce la faute du SMOTE ?         (Cause A)
  Q2 — Est-ce la faute des features ?     (Cause B)
  Q3 — Est-ce la faute du modèle RF ?     (Cause C)

Chaque section imprime un verdict clair.

Usage :
    python src/diagnostic_compression.py
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import roc_auc_score, brier_score_loss
from sklearn.calibration     import calibration_curve
from imblearn.over_sampling  import SMOTE

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN    = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR    = os.path.join(BASE_DIR, 'reports', 'diagnostic')
os.makedirs(OUT_DIR, exist_ok=True)

SEP = "=" * 65

LEAKAGE_CLF = [
    'ChurnRiskCategory', 'CustomerType', 'AccountStatus',
    'Recency', 'TenureRatio', 'MonetaryPerDay',
    'FirstPurchaseDaysAgo', 'CustomerTenureDays', 'LoyaltyLevel',
    'RFMSegment', 'rfm_cluster',
    'rfm_seg_Clients_Champions', 'rfm_seg_Clients_Fideles',
    'rfm_seg_Clients_Perdus',    'rfm_seg_Clients_a_Risque',
]


# ══════════════════════════════════════════════════════════════
# CHARGEMENT & PREPARATION
# ══════════════════════════════════════════════════════════════

def charger_et_preparer():
    print(f"\n{SEP}")
    print("  CHARGEMENT DES DONNÉES")
    print(SEP)

    df = pd.read_csv(DATA_IN)
    print(f"  Shape : {df.shape}")
    print(f"  Churn : {df['Churn'].value_counts().to_dict()}")
    print(f"  Taux churn : {df['Churn'].mean()*100:.1f}%")

    # Nettoyage identique à train_model.py
    if 'MonetaryTotal' in df.columns:
        df.loc[df['MonetaryTotal'] < 0, 'MonetaryTotal'] = np.nan
    if 'RFMSegment' in df.columns:
        df = df.drop(columns=['RFMSegment'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split identique à train_model.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Suppression leakage
    supprimees = [c for c in LEAKAGE_CLF if c in X_train.columns]
    X_train = X_train.drop(columns=supprimees)
    X_test  = X_test.drop(columns=supprimees)

    # Target encoding Country
    if 'Country' in X_train.columns:
        enc = joblib.load(os.path.join(MODELS_DIR, 'country_encoding.pkl'))
        X_train['Country_encoded'] = X_train['Country'].map(enc['mapping']).fillna(enc['moy_globale'])
        X_test['Country_encoded']  = X_test['Country'].map(enc['mapping']).fillna(enc['moy_globale'])
        X_train = X_train.drop(columns=['Country'])
        X_test  = X_test.drop(columns=['Country'])

    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

    return X_train_imp, X_test_imp, y_train, y_test


# ══════════════════════════════════════════════════════════════
# Q1 — EST-CE LA FAUTE DU SMOTE ?
# ══════════════════════════════════════════════════════════════

def diagnostiquer_smote(X_train, X_test, y_train, y_test):
    print(f"\n{SEP}")
    print("  Q1 — IMPACT DU SMOTE SUR LES PROBABILITÉS")
    print(SEP)

    resultats = {}

    # ── Modèle A : avec SMOTE (reproduction de train_model.py) ──
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    rf_smote = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    rf_smote.fit(X_bal, y_bal)
    proba_smote = rf_smote.predict_proba(X_test)[:, 1]

    # ── Modèle B : SANS SMOTE, class_weight='balanced' ──────────
    rf_balanced = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_balanced.fit(X_train, y_train)
    proba_balanced = rf_balanced.predict_proba(X_test)[:, 1]

    # ── Modèle C : SANS SMOTE, sans pondération ─────────────────
    rf_brut = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    rf_brut.fit(X_train, y_train)
    proba_brut = rf_brut.predict_proba(X_test)[:, 1]

    for nom, proba in [('SMOTE', proba_smote),
                       ('class_weight=balanced', proba_balanced),
                       ('Brut (sans rien)', proba_brut)]:
        auc    = roc_auc_score(y_test, proba)
        brier  = brier_score_loss(y_test, proba)
        p_min  = proba.min()
        p_max  = proba.max()
        p_std  = proba.std()
        resultats[nom] = {'auc': auc, 'brier': brier, 'min': p_min, 'max': p_max, 'std': p_std}

        print(f"\n  [{nom}]")
        print(f"    AUC-ROC  : {auc:.4f}")
        print(f"    Brier    : {brier:.4f}  (plus bas = mieux calibré)")
        print(f"    Plage    : [{p_min:.3f} — {p_max:.3f}]")
        print(f"    Écart-type proba : {p_std:.4f}  (plus élevé = mieux étalé)")

    # ── Verdict ──────────────────────────────────────────────────
    std_smote    = resultats['SMOTE']['std']
    std_balanced = resultats['class_weight=balanced']['std']
    std_brut     = resultats['Brut (sans rien)']['std']

    print(f"\n  VERDICT Q1 :")
    if std_balanced > std_smote * 1.15 or std_brut > std_smote * 1.15:
        print("  → SMOTE EST COUPABLE : les modèles sans SMOTE ont des")
        print("    probabilités plus étalées. La solution est de remplacer")
        print("    SMOTE par class_weight='balanced'.")
        coupable_smote = True
    else:
        print("  → SMOTE n'est PAS la cause principale : les 3 modèles")
        print("    ont des plages similaires. Chercher ailleurs.")
        coupable_smote = False

    # ── Courbe de calibration ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Calibration parfaite')
    couleurs = {'SMOTE': '#E24B4A', 'class_weight=balanced': '#1D9E75', 'Brut (sans rien)': '#378ADD'}
    for nom, proba in [('SMOTE', proba_smote),
                       ('class_weight=balanced', proba_balanced),
                       ('Brut (sans rien)', proba_brut)]:
        fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker='o', label=nom, color=couleurs[nom])

    ax.set_xlabel('Probabilité prédite moyenne')
    ax.set_ylabel('Fraction réelle de positifs')
    ax.set_title('Courbes de calibration — SMOTE vs alternatives')
    ax.legend()
    ax.grid(True, alpha=0.3)
    chemin = os.path.join(OUT_DIR, 'q1_calibration_smote.png')
    plt.tight_layout()
    plt.savefig(chemin, dpi=120)
    plt.close()
    print(f"\n  Courbe sauvegardée : {chemin}")

    return coupable_smote, resultats


# ══════════════════════════════════════════════════════════════
# Q2 — EST-CE LA FAUTE DES FEATURES ?
# ══════════════════════════════════════════════════════════════

def diagnostiquer_features(X_train, X_test, y_train, y_test):
    print(f"\n{SEP}")
    print("  Q2 — QUALITÉ DU SIGNAL DANS LES FEATURES")
    print(SEP)

    # Charger le modèle de production pour les importances réelles
    try:
        rf_prod = joblib.load(os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
        importances = pd.Series(
            rf_prod.feature_importances_,
            index=joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        ).sort_values(ascending=False)
    except Exception:
        # Fallback : recalculer
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        rf_prod = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_prod.fit(X_bal, y_bal)
        importances = pd.Series(rf_prod.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    top5_pct  = importances.head(5).sum() * 100
    top10_pct = importances.head(10).sum() * 100
    top20_pct = importances.head(20).sum() * 100

    print(f"\n  Concentration du signal :")
    print(f"    Top  5 features : {top5_pct:.1f}% de l'importance totale")
    print(f"    Top 10 features : {top10_pct:.1f}% de l'importance totale")
    print(f"    Top 20 features : {top20_pct:.1f}% de l'importance totale")

    print(f"\n  Top 15 features :")
    for feat, imp in importances.head(15).items():
        barre = '█' * int(imp * 300)
        print(f"    {feat:<35} {imp:.4f}  {barre}")

    # Features avec importance quasi nulle
    mortes = importances[importances < 0.002]
    print(f"\n  Features quasi-inutiles (importance < 0.002) : {len(mortes)}")
    if len(mortes) > 0:
        print(f"    {list(mortes.index[:10])}")

    # ── Test : le modèle avec seulement les top features ─────────
    top_feats = importances.head(10).index.tolist()
    smote = SMOTE(random_state=42)
    X_bal_top, y_bal = smote.fit_resample(X_train[top_feats], y_train)
    rf_top = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_top.fit(X_bal_top, y_bal)
    proba_top = rf_top.predict_proba(X_test[top_feats])[:, 1]
    auc_top   = roc_auc_score(y_test, proba_top)

    # Modèle complet pour comparaison
    X_bal_all, _ = SMOTE(random_state=42).fit_resample(X_train, y_train)
    rf_all = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_all.fit(X_bal_all, y_bal)
    proba_all = rf_all.predict_proba(X_test)[:, 1]
    auc_all   = roc_auc_score(y_test, proba_all)

    print(f"\n  AUC modèle complet ({X_train.shape[1]} features) : {auc_all:.4f}")
    print(f"  AUC top-10 features seulement              : {auc_top:.4f}")
    print(f"  Plage proba top-10 : [{proba_top.min():.3f} — {proba_top.max():.3f}]  std={proba_top.std():.4f}")
    print(f"  Plage proba complet: [{proba_all.min():.3f} — {proba_all.max():.3f}]  std={proba_all.std():.4f}")

    # ── Verdict ──────────────────────────────────────────────────
    print(f"\n  VERDICT Q2 :")
    signal_faible = False
    if top5_pct < 55:
        print("  → SIGNAL DILUÉ : les top 5 features représentent moins de 55%")
        print("    de l'importance. Le modèle s'appuie sur beaucoup de features")
        print("    faibles, ce qui comprime les probabilités vers 0.5.")
        signal_faible = True
    else:
        print(f"  → Signal OK : top 5 = {top5_pct:.0f}%. La dilution des features")
        print("    n'est probablement pas la cause principale.")

    if len(mortes) > 10:
        print(f"  → {len(mortes)} features mortes détectées. Les supprimer peut aider.")

    # Graphique importances
    fig, ax = plt.subplots(figsize=(9, 6))
    top20 = importances.head(20)
    ax.barh(top20.index[::-1], top20.values[::-1], color='#378ADD', alpha=0.8)
    ax.set_xlabel('Importance (Gini)')
    ax.set_title('Top 20 feature importances — modèle de production')
    ax.grid(True, axis='x', alpha=0.3)
    chemin = os.path.join(OUT_DIR, 'q2_feature_importances.png')
    plt.tight_layout()
    plt.savefig(chemin, dpi=120)
    plt.close()
    print(f"\n  Graphique sauvegardé : {chemin}")

    return signal_faible, importances


# ══════════════════════════════════════════════════════════════
# Q3 — EST-CE LA FAUTE DU MODÈLE RF ?
# ══════════════════════════════════════════════════════════════

def diagnostiquer_modele(X_train, X_test, y_train, y_test):
    print(f"\n{SEP}")
    print("  Q3 — CONTRAINTES DU RANDOM FOREST")
    print(SEP)

    # Paramètres du modèle de production
    try:
        rf_prod    = joblib.load(os.path.join(MODELS_DIR, 'random_forest_churn.pkl'))
        params     = rf_prod.get_params()
        max_depth  = params.get('max_depth', '?')
        n_est      = params.get('n_estimators', '?')
        min_split  = params.get('min_samples_split', '?')
        print(f"\n  Paramètres actuels du modèle de production :")
        print(f"    max_depth          : {max_depth}")
        print(f"    n_estimators       : {n_est}")
        print(f"    min_samples_split  : {min_split}")
    except Exception:
        print("  Impossible de charger le modèle de production.")
        max_depth = 10

    # Test avec différentes profondeurs
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    print(f"\n  Impact de max_depth sur l'étalement des probabilités :")
    print(f"  {'max_depth':<12} {'AUC':<8} {'Plage proba':<20} {'Std proba':<10}")
    print(f"  {'-'*55}")

    resultats_depth = {}
    for depth in [5, 10, 20, None]:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=depth,
            min_samples_split=2, random_state=42, n_jobs=-1
        )
        rf.fit(X_bal, y_bal)
        proba = rf.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        std   = proba.std()
        label = str(depth) if depth else 'None (illimité)'
        plage = f"[{proba.min():.2f}–{proba.max():.2f}]"
        print(f"  {label:<12} {auc:.4f}   {plage:<20} {std:.4f}")
        resultats_depth[label] = {'auc': auc, 'std': std, 'proba': proba}

    # ── Verdict ──────────────────────────────────────────────────
    std_depth5    = resultats_depth.get('5', {}).get('std', 0)
    std_depth_inf = resultats_depth.get('None (illimité)', {}).get('std', 0)

    print(f"\n  VERDICT Q3 :")
    modele_contraint = False
    if std_depth_inf > std_depth5 * 1.2:
        print("  → MAX_DEPTH TROP FAIBLE : les arbres plus profonds produisent")
        print("    des probabilités plus étalées. Augmenter max_depth dans la grille.")
        modele_contraint = True
    else:
        print("  → La profondeur des arbres n'est pas le problème principal.")
        print("    La compression vient d'ailleurs.")

    return modele_contraint, resultats_depth


# ══════════════════════════════════════════════════════════════
# SYNTHÈSE FINALE
# ══════════════════════════════════════════════════════════════

def synthese(coupable_smote, signal_faible, modele_contraint):
    print(f"\n{SEP}")
    print("  SYNTHÈSE — PLAN D'ACTION RECOMMANDÉ")
    print(SEP)

    causes = []
    if coupable_smote:    causes.append("SMOTE (Cause A)")
    if signal_faible:     causes.append("Features faibles (Cause B)")
    if modele_contraint:  causes.append("RF trop contraint (Cause C)")

    if not causes:
        print("\n  Aucune cause claire isolée. La compression est probablement")
        print("  due à la nature des données (churn difficile à distinguer).")
        print("  Envisager : GradientBoosting ou LightGBM à la place du RF.")
        return

    print(f"\n  Causes identifiées : {', '.join(causes)}")
    print()

    if coupable_smote:
        print("  [PRIORITÉ 1] Remplacer SMOTE dans train_model.py :")
        print("    AVANT : smote = SMOTE(random_state=42)")
        print("            X_bal, y_bal = smote.fit_resample(X_train, y_train)")
        print("    APRÈS : Passer class_weight='balanced' au RandomForestClassifier")
        print("            et supprimer entièrement l'étape SMOTE.")
        print("    POURQUOI : class_weight pénalise les erreurs sur la classe")
        print("    minoritaire SANS créer de faux exemples synthétiques qui")
        print("    brouillent la frontière de décision.")
        print()

    if signal_faible:
        print("  [PRIORITÉ 2] Enrichir les features dans data_clean.csv :")
        print("    Ajouter : DaysSinceLastOrder (= Recency calculée côté features,")
        print("              pas comme variable de leakage)")
        print("    Ajouter : FrequencyTrend (fréquence récente vs ancienne)")
        print("    Ajouter : MonetaryTrend  (dépense récente vs ancienne)")
        print("    Supprimer : les features avec importance < 0.002")
        print()

    if modele_contraint:
        print("  [PRIORITÉ 3] Élargir la grille de recherche :")
        print("    param_grid = {")
        print("        'max_depth'        : [10, 20, 30, None],")
        print("        'n_estimators'     : [200, 300, 500],")
        print("        'min_samples_split': [2, 5, 10],")
        print("        'min_samples_leaf' : [1, 2, 4],")
        print("    }")
        print()

    print("  ORDRE D'EXÉCUTION RECOMMANDÉ :")
    print("    1. Appliquer les corrections dans train_model.py")
    print("    2. Relancer l'entraînement complet")
    print("    3. Retester avec les 4 cas de test")
    print("    4. Si prob Cas1 < 20% et Cas2 > 75% → problème résolu")
    print("    5. Sinon → relancer ce script pour revalider")

    print(f"\n  Rapports sauvegardés dans : reports/diagnostic/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(SEP)
    print("  DIAGNOSTIC — COMPRESSION DES PROBABILITÉS")
    print(SEP)

    X_train, X_test, y_train, y_test = charger_et_preparer()

    coupable_smote,   _  = diagnostiquer_smote(X_train, X_test, y_train, y_test)
    signal_faible,    _  = diagnostiquer_features(X_train, X_test, y_train, y_test)
    modele_contraint, _  = diagnostiquer_modele(X_train, X_test, y_train, y_test)

    synthese(coupable_smote, signal_faible, modele_contraint)