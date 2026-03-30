import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'C:\Users\abdal\Desktop\projects\ML\ML_project\src')

# On importe les fonctions de détection depuis utils.py
from utils import (
    analyser_uniques,       # détecte les colonnes inutiles
    analyser_redondantes,   # détecte les colonnes redondantes
    # analyser_manquants,     # détecte les valeurs manquantes
)

# ============================================================
# FONCTION 1 : Supprimer les colonnes inutiles
# ============================================================
def supprimer_colonnes_inutiles(df):
    """
    Utilise utils.py pour détecter les colonnes inutiles
    puis les supprime
    """
    print("\n  SUPPRESSION DES COLONNES INUTILES")

    # utils.py détecte automatiquement les colonnes avec 1 seule valeur
    colonnes_inutiles = analyser_uniques(df)

    # On ajoute CustomerID manuellement car c'est un identifiant
    # (pas détecté par analyser_uniques car il a beaucoup de valeurs uniques)
    if 'CustomerID' not in colonnes_inutiles:
        colonnes_inutiles.append('CustomerID')

    # preprocessing.py supprime
    for col in colonnes_inutiles:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f" {col} supprimée")
    features_leakage = [
      'ChurnRiskCategory',   # ❌ leakage confirmé
      'CustomerType',
      'AccountStatus',
      'RFMSegment'
]
    df = df.drop(columns=[c for c in features_leakage if c in df.columns])


    return df


# ============================================================
# FONCTION 2 : Supprimer les colonnes redondantes
# ============================================================
def supprimer_redondantes(df):
    """
    Utilise utils.py pour détecter les colonnes redondantes
    puis les supprime
    """
    print("\n SUPPRESSION DES COLONNES REDONDANTES")

    # utils.py détecte automatiquement les colonnes corrélées > 0.8
    colonnes_redondantes = analyser_redondantes(df, seuil=0.8)

    # preprocessing.py supprime
    for col in colonnes_redondantes:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f" {col} supprimée")

    print(f"Shape après suppression : {df.shape}")
    return df


# ============================================================
# FONCTION 3 : Corriger les valeurs aberrantes
# ============================================================
def corriger_aberrantes(df):
    """
    Utilise utils.analyser_aberrantes() pour détecter
    puis corrige selon les règles métier du PDF page 4
    """
    print("\n  CORRECTION DES VALEURS ABERRANTES")
    print("    → NaN sera imputé dans train_model.py après le split")

    # preprocessing.py corrige selon règles métier
    # SatisfactionScore : valides = 1 à 5 (-1, 0, 99 = aberrants)
    nb = df[~df['SatisfactionScore'].between(1, 5)].shape[0]
    df.loc[~df['SatisfactionScore'].between(1, 5), 'SatisfactionScore'] = np.nan
    print(f" SatisfactionScore : {nb} aberrantes → NaN")

    # SupportTicketsCount : valides = 0 à 15 (-1, 999 = aberrants)
    nb = df[~df['SupportTicketsCount'].between(0, 15)].shape[0]
    df.loc[~df['SupportTicketsCount'].between(0, 15), 'SupportTicketsCount'] = np.nan
    print(f" SupportTicketsCount : {nb} aberrantes → NaN")

    return df

# ============================================================
# FONCTION 4 : Feature Engineering (PDF page 6)
# ============================================================
def feature_engineering(df):
    """
    Création de nouvelles features (PDF page 6)
    + Parsing RegistrationDate (PDF page 7)
    + Extraction LastLoginIP (PDF page 7)
    """
    print("\n FEATURE ENGINEERING")

    # ── Nouvelles features (PDF page 6) ──
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    print(" MonetaryPerDay = MonetaryTotal / (Recency + 1)")

    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    print(" AvgBasketValue = MonetaryTotal / Frequency")

    # PDF utilise CustomerTenure → notre CSV = CustomerTenureDays
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print(" TenureRatio = Recency / CustomerTenureDays")

    # ── Parsing RegistrationDate  ──
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            dayfirst=True,   # priorité format UK
            errors='coerce'  # NaT si format inconnu
        )
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])
        print(" RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")

    # ── Extraction LastLoginIP ──
    if 'LastLoginIP' in df.columns:
        # Premier octet → indice géographique
        df['IP_PremierOctet'] = df['LastLoginIP'].apply(
            lambda x: int(str(x).split('.')[0]) if pd.notna(x) else 0
        )
        # IP privée vs publique (PDF : détecter si IP privée/publique)
        df['IP_Privee'] = df['LastLoginIP'].apply(
            lambda x: 1 if str(x).startswith(('10.', '192.168.', '172.'))
            else 0 if pd.notna(x) else 0
        )
        df = df.drop(columns=['LastLoginIP'])
        print(" LastLoginIP → IP_PremierOctet, IP_Privee")

    print(f"Shape après feature engineering : {df.shape}")
    return df

# ============================================================
# FONCTION 7 : Encoder les colonnes texte
# ============================================================
def encoder_colonnes(df):
    """
    Encodage des colonnes texte (PDF pages 3-4)
    - Ordinal  : quand il y a un ordre logique
    - One-Hot  : quand il n'y a pas d'ordre
    - Country  : Target Encoding → train_model.py après split
    """
    print("\n ENCODAGE DES COLONNES TEXTE")

    # ── Encodage Ordinal  ──
    encodages_ordinaux = {
        # ordre : Low < Medium < High < VIP
        'SpendingCategory'  : ['Low', 'Medium', 'High', 'VIP'],
        
        # ordre : Nouveau < Jeune < Établi < Ancien (Inconnu = -1)
        'LoyaltyLevel'      : ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        
        # ordre : Faible < Moyen < Élevé < Critique
        'ChurnRiskCategory' : ['Faible', 'Moyen', 'Élevé', 'Critique'],
        
        # ordre : Petit < Moyen < Grand (Inconnu = -1)
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        
        # ordre chronologique des tranches d'âge
        'AgeCategory'       : ['18-24', '25-34', '35-44', '45-54',
                               '55-64', '65+', 'Inconnu'],
        
        # ordre chronologique de la journée
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
        
        # ordre valeur client : Dormants < Potentiels < Fidèles < Champions
        # PDF dit Ord/One-Hot → on choisit Ordinal car ordre logique prouvé
        # (Dormants=100% churn, Champions=0% churn)
        'RFMSegment'        : ['Dormants', 'Potentiels', 'Fidèles', 'Champions'],
    }

    for col, ordre in encodages_ordinaux.items():
        if col in df.columns:
            df[col] = pd.Categorical(
                df[col], categories=ordre, ordered=True).codes
            print(f" {col} encodé (ordinal)")

    # ── Encodage One-Hot (pas d'ordre → PDF page 3) ──
    colonnes_onehot = [
        'CustomerType',      # One-Hot (PDF) → pas d'ordre entre profils
        'FavoriteSeason',    # One-Hot (PDF) → pas d'ordre entre saisons
        'Region',            # One-Hot (PDF) → pas d'ordre entre régions
        'WeekendPreference', # One-Hot (PDF) → pas d'ordre
        'ProductDiversity',  # One-Hot (PDF) → pas d'ordre
        'Gender',            # One-Hot (PDF) → pas d'ordre
        'AccountStatus',     # One-Hot (PDF) → pas d'ordre
    ]

    colonnes_onehot = [c for c in colonnes_onehot if c in df.columns]
    df = pd.get_dummies(df, columns=colonnes_onehot, drop_first=True)
    print(f" One-Hot appliqué sur : {colonnes_onehot}")

    # ── Country → Target Encoding dans train_model.py ──
    # Gardé tel quel → Target Encoding après split pour éviter leakage
    if 'Country' in df.columns:
        print(" Country gardé → Target Encoding dans train_model.py après split")

    return df

# ============================================================
# PIPELINE COMPLET
# ============================================================
def pipeline_complet(df):
    """Lance tout le preprocessing d'un coup"""
    print(" Démarrage du preprocessing...\n")

    # utils.py détecte → preprocessing.py supprime/corrige
    df = supprimer_colonnes_inutiles(df)  # utils détecte les inutiles
    df = supprimer_redondantes(df)        # utils détecte les redondantes
    df = corriger_aberrantes(df)          # utils détecte les aberrantes
    df = feature_engineering(df)          # utils ne fait pas de feature engineering
    df = encoder_colonnes(df)

    print(f"\n Preprocessing terminé !")
    print(f"Shape final : {df.shape}")

    return df


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\abdal\Desktop\projects\ML\ML_project\data\raw\retail_customers_COMPLETE CATEGORICAL.csv')
    print(" Données chargées !")

    df_clean = pipeline_complet(df)

    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/data_clean.csv', index=False)
    print("\n Données sauvegardées dans data/processed/data_clean.csv")
    print("  Colonnes avec NaN restants (imputés dans train_model.py) :")
    manquants = df_clean.isnull().sum()
    print(manquants[manquants > 0])