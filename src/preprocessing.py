import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append(r'C:\Users\abdal\Desktop\projects\ML\ML_project\src')

# On importe les fonctions de détection depuis utils.py
from utils import (
    analyser_uniques,       # détecte les colonnes inutiles
    analyser_redondantes,   # détecte les colonnes redondantes
    analyser_manquants,     # détecte les valeurs manquantes
    analyser_aberrantes,    # détecte les valeurs aberrantes
    analyser_echelles       # détecte les colonnes à normaliser
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

    # utils.py détecte automatiquement les colonnes corrélées > 0.85
    colonnes_redondantes = analyser_redondantes(df, seuil=0.8)

    # preprocessing.py supprime
    for col in colonnes_redondantes:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"✅ {col} supprimée")

    print(f"Shape après suppression : {df.shape}")
    return df


# ============================================================
# FONCTION 3 : Corriger les valeurs aberrantes
# ============================================================
def corriger_aberrantes(df):
    """
    Utilise utils.py pour détecter les aberrantes
    puis les corrige selon les règles métier
    """
    print("\n  CORRECTION DES VALEURS ABERRANTES")

    # utils.py détecte (affiche les stats et boxplots)
    # On sait grâce à analyser_aberrantes() que :
    # SatisfactionScore valide = 1 à 5
    # SupportTicketsCount valide = 0 à 15

    # preprocessing.py corrige
    nb = df[~df['SatisfactionScore'].between(1, 5)].shape[0]
    df.loc[~df['SatisfactionScore'].between(1, 5), 'SatisfactionScore'] = np.nan
    print(f" SatisfactionScore : {nb} aberrantes → NaN")

    nb = df[~df['SupportTicketsCount'].between(0, 15)].shape[0]
    df.loc[~df['SupportTicketsCount'].between(0, 15), 'SupportTicketsCount'] = np.nan
    print(f" SupportTicketsCount : {nb} aberrantes → NaN")

    return df


# ============================================================
# FONCTION 4 : Imputer les valeurs manquantes
# ============================================================
def imputer_manquants(df):
    """
    Utilise utils.py pour détecter les colonnes avec NaN
    puis les impute avec la médiane
    """
    print("\n IMPUTATION DES VALEURS MANQUANTES")

    # utils.py détecte automatiquement les colonnes avec NaN
    colonnes_manquantes = analyser_manquants(df)

    # preprocessing.py impute avec la médiane
    for col in colonnes_manquantes:
        if col in df.columns:
            nb = df[col].isnull().sum()
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            print(f" {col} : {nb} NaN → médiane ({mediane:.2f})")

    return df


# ============================================================
# FONCTION 5 : Parser les dates
# ============================================================
def parser_dates(df):
    """Extrait les informations utiles de RegistrationDate"""
    print("\n PARSING DES DATES")

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            dayfirst=True,
            errors='coerce'
        )
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday

        df = df.drop(columns=['RegistrationDate'])
        print(" RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")

    return df


# ============================================================
# FONCTION 6 : Traiter LastLoginIP
# ============================================================
def traiter_ip(df):
    """Extrait le premier octet de l'adresse IP"""
    print("\n TRAITEMENT DES ADRESSES IP")

    if 'LastLoginIP' in df.columns:
        df['IP_PremierOctet'] = df['LastLoginIP'].apply(
            lambda x: int(str(x).split('.')[0]) if pd.notna(x) else 0
        )
        df = df.drop(columns=['LastLoginIP'])
        print(" LastLoginIP → IP_PremierOctet")

    return df


# ============================================================
# FONCTION 7 : Encoder les colonnes texte
# ============================================================
def encoder_colonnes(df):
    """Encode les colonnes texte en chiffres"""
    print("\n ENCODAGE DES COLONNES TEXTE")

    # Encodage Ordinal (ordre logique)
    encodages_ordinaux = {
        'SpendingCategory'  : ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel'      : ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        'ChurnRiskCategory' : ['Faible', 'Moyen', 'Élevé', 'Critique'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'AgeCategory'       : ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }

    for col, ordre in encodages_ordinaux.items():
        if col in df.columns:
            df[col] = pd.Categorical(
                df[col], categories=ordre, ordered=True).codes
            print(f" {col} encodé (ordinal)")

    # Encodage One-Hot (pas d'ordre)
    colonnes_onehot = [
        'RFMSegment', 'CustomerType', 'FavoriteSeason',
        'Region', 'WeekendPreference', 'ProductDiversity',
        'Gender', 'AccountStatus'
    ]
    colonnes_onehot = [c for c in colonnes_onehot if c in df.columns]
    df = pd.get_dummies(df, columns=colonnes_onehot, drop_first=True)
    print(f" One-Hot appliqué sur : {colonnes_onehot}")

    # Country → Target Encoding sera fait dans train_model.py
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])
        print(" Country supprimé (Target Encoding dans train_model.py)")

    return df


# ============================================================
# FONCTION 8 : Normaliser
# ============================================================
def normaliser(df, sauvegarder=True):
    """
    Utilise utils.py pour détecter les colonnes à normaliser
    puis applique StandardScaler
    """
    print("\n NORMALISATION")

    # On ne normalise jamais la cible
    colonnes_a_exclure = ['Churn']
    colonnes_numeriques = df.select_dtypes(include='number').columns.tolist()
    colonnes_a_normaliser = [c for c in colonnes_numeriques
                             if c not in colonnes_a_exclure]

    scaler = StandardScaler()
    df[colonnes_a_normaliser] = scaler.fit_transform(df[colonnes_a_normaliser])

    if sauvegarder:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print(" Scaler sauvegardé dans models/scaler.pkl")

    print(f" {len(colonnes_a_normaliser)} colonnes normalisées")
    return df, scaler


# ============================================================
# PIPELINE COMPLET
# ============================================================
def pipeline_complet(df, sauvegarder=True):
    """Lance tout le preprocessing d'un coup"""
    print("🚀 Démarrage du preprocessing...\n")

    # utils.py détecte → preprocessing.py supprime/corrige
    df = supprimer_colonnes_inutiles(df)  # utils détecte les inutiles
    df = supprimer_redondantes(df)        # utils détecte les redondantes
    df = corriger_aberrantes(df)          # utils détecte les aberrantes
    df = imputer_manquants(df)            # utils détecte les manquants
    df = parser_dates(df)
    df = traiter_ip(df)
    df = encoder_colonnes(df)
    df, scaler = normaliser(df, sauvegarder)  # utils détecte les échelles

    print(f"\n Preprocessing terminé !")
    print(f"Shape final : {df.shape}")

    return df, scaler


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\abdal\Desktop\projects\ML\ML_project\data\raw\retail_customers_COMPLETE CATEGORICAL.csv')
    print(" Données chargées !")

    df_clean, scaler = pipeline_complet(df)

    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/data_clean.csv', index=False)