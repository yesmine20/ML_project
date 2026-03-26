#  le nettoyage spécifique à CE projet (imputer l'âge, encoder les catégories, normaliser...)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# FONCTION 1 : Supprimer les colonnes inutiles

def supprimer_colonnes_inutiles(df):
    """Supprime les colonnes qui n'apportent aucune information"""
    colonnes_a_supprimer = ['NewsletterSubscribed', 'CustomerID']
    
    for col in colonnes_a_supprimer:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"✅ Colonne supprimée : {col}")
    
    return df


# 
# FONCTION 2 : Corriger les valeurs aberrantes
# 
def corriger_aberrantes(df):
    """Remplace les valeurs aberrantes par NaN"""
    print("\n⚠️  Correction des valeurs aberrantes...")
    
    # SatisfactionScore : valeurs valides = 1 à 5
    nb = df[~df['SatisfactionScore'].between(1, 5)].shape[0]
    df.loc[~df['SatisfactionScore'].between(1, 5), 'SatisfactionScore'] = np.nan
    print(f"✅ SatisfactionScore : {nb} valeurs aberrantes remplacées par NaN")
    
    # SupportTicketsCount : valeurs valides = 0 à 15
    nb = df[~df['SupportTicketsCount'].between(0, 15)].shape[0]
    df.loc[~df['SupportTicketsCount'].between(0, 15), 'SupportTicketsCount'] = np.nan
    print(f"✅ SupportTicketsCount : {nb} valeurs aberrantes remplacées par NaN")
    
    return df

# FONCTION 3 : Imputer les valeurs manquantes

def imputer_manquants(df):
    """Remplace les valeurs manquantes par la médiane"""
    print("\n🔧 Imputation des valeurs manquantes...")
    
    colonnes_a_imputer = [
        'Age',
        'AvgDaysBetweenPurchases',
        'SatisfactionScore',
        'SupportTicketsCount'
    ]
    
    for col in colonnes_a_imputer:
        if col in df.columns:
            nb = df[col].isnull().sum()
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            print(f"✅ {col} : {nb} manquants remplacés par médiane ({mediane:.2f})")
    
    return df

# FONCTION 4 : Parser RegistrationDate

def parser_dates(df):
    """Extrait les informations utiles de RegistrationDate"""
    print("\n📅 Parsing des dates...")
    
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
        print("✅ RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")
    
    return df

# FONCTION 5 : Traiter LastLoginIP
def traiter_ip(df):
    """Extrait le premier octet de l'adresse IP"""
    print("\n🌐 Traitement des adresses IP...")
    
    if 'LastLoginIP' in df.columns:
        df['IP_PremierOctet'] = df['LastLoginIP'].apply(
            lambda x: int(str(x).split('.')[0]) if pd.notna(x) else 0
        )
        df = df.drop(columns=['LastLoginIP'])
        print("✅ LastLoginIP → IP_PremierOctet")
    
    return df

# FONCTION 6 : Encoder les colonnes texte

def encoder_colonnes(df):
    """Encode les colonnes texte en chiffres"""
    print("\n🔤 Encodage des colonnes texte...")
    
    # Encodage Ordinal (ordre logique)
    encodages_ordinaux = {
        'SpendingCategory' : ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel'     : ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        'ChurnRiskCategory': ['Faible', 'Moyen', 'Élevé', 'Critique'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'AgeCategory'      : ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }
    
    for col, ordre in encodages_ordinaux.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=ordre, ordered=True).codes
            print(f"✅ {col} encodé (ordinal)")
    
    # Encodage One-Hot (pas d'ordre)
    colonnes_onehot = [
        'RFMSegment', 'CustomerType', 'FavoriteSeason',
        'Region', 'WeekendPreference', 'ProductDiversity',
        'Gender', 'AccountStatus'
    ]
    
    colonnes_onehot = [c for c in colonnes_onehot if c in df.columns]
    df = pd.get_dummies(df, columns=colonnes_onehot, drop_first=True)
    print(f"✅ One-Hot encoding appliqué sur : {colonnes_onehot}")
    
    # Supprimer Country (trop de valeurs uniques)
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])
        print("✅ Country supprimé (trop de valeurs uniques)")
    
    return df

# FONCTION 7 : Normaliser les colonnes numériques

def normaliser(df, sauvegarder=True):
    """Normalise les colonnes numériques avec StandardScaler"""
    print("\n📏 Normalisation...")
    
    # On ne normalise pas la cible Churn
    colonnes_a_exclure = ['Churn']
    colonnes_numeriques = df.select_dtypes(include='number').columns.tolist()
    colonnes_a_normaliser = [c for c in colonnes_numeriques 
                             if c not in colonnes_a_exclure]
    
    scaler = StandardScaler()
    df[colonnes_a_normaliser] = scaler.fit_transform(df[colonnes_a_normaliser])
    
    if sauvegarder:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✅ Scaler sauvegardé dans models/scaler.pkl")
    
    print(f"✅ {len(colonnes_a_normaliser)} colonnes normalisées")
    return df, scaler



# PIPELINE COMPLET

def pipeline_complet(df, sauvegarder=True):
    """Lance tout le preprocessing d'un coup"""
    print("🚀 Démarrage du preprocessing...\n")
    
    df = supprimer_colonnes_inutiles(df)
    df = corriger_aberrantes(df)
    df = imputer_manquants(df)
    df = parser_dates(df)
    df = traiter_ip(df)
    df = encoder_colonnes(df)
    df, scaler = normaliser(df, sauvegarder)
    
    print(f"\n✅ Preprocessing terminé !")
    print(f"Shape final : {df.shape}")
    
    return df, scaler


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\abdal\Desktop\projects\ML\ML_project\data\raw\retail_customers_COMPLETE CATEGORICAL.csv')
    print("✅ Données chargées !")
    
    df_clean, scaler = pipeline_complet(df)
    
    # Sauvegarder les données nettoyées
    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/data_clean.csv', index=False)
    print("\n✅ Données nettoyées sauvegardées dans data/processed/data_clean.csv")


# ## Résumé de ce que fait chaque fonction
# ```
# pipeline_complet(df)
#         │
#         ├── supprimer_colonnes_inutiles() → supprime NewsletterSubscribed, CustomerID
#         ├── corriger_aberrantes()         → corrige 999, -1 → NaN
#         ├── imputer_manquants()           → remplace NaN par médiane
#         ├── parser_dates()               → RegistrationDate → 4 colonnes
#         ├── traiter_ip()                 → LastLoginIP → IP_PremierOctet
#         ├── encoder_colonnes()           → texte → chiffres
#         └── normaliser()                 → même échelle + sauvegarde scaler.pkl