from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
sys.path.append(r'C:\Users\abdal\Desktop\projects\ML\ML_project\src')
from sklearn.feature_selection import VarianceThreshold
import pandas as pd


# FONCTION 1 : Supprimer les colonnes inutiles (Variance nulle et quasi-nulle)
def supprimer_colonnes_inutiles(df, seuil_variance=0.01, seuil_frequence_texte=0.95):
    """
    Supprime les colonnes constantes et quasi-constantes.
    - Utilise VarianceThreshold (scikit-learn) pour les variables numériques.
    - Utilise la fréquence (>95% identiques) pour les variables textuelles.
    """
    print(f"Shape avant suppression : {df.shape}")
    print("Recherche des colonnes à variance nulle ou quasi-nulle...")
    
    colonnes_a_supprimer = ['CustomerID']  # CustomerID est souvent inutile pour la modélisation (identifiant unique)
    
    #Séparer les colonnes numériques et textuelles
    cols_num = df.select_dtypes(include=['number']).columns
    cols_cat = df.select_dtypes(exclude=['number']).columns
    
    #numériques +VarinceThreshold
    if len(cols_num) > 0:
        # Initialisation du filtre avec un seuil (ex: 0.01 supprime les quasi-constantes)
        selector = VarianceThreshold(threshold=seuil_variance)
        selector.fit(df[cols_num]) # Apprend quelles colonnes garder
        
        # selector.get_support() renvoie True (garder) ou False (supprimer)
        cols_num_gardees = cols_num[selector.get_support()]
        cols_num_jetees = [c for c in cols_num if c not in cols_num_gardees]
        # print(f" Colonnes numériques à supprimer (variance < {seuil_variance}) : {cols_num_jetees}")

        colonnes_a_supprimer.extend(cols_num_jetees)
        
    # variables TEXTUELLES (Catégorielles)
    for col in cols_cat:
        # On regarde la fréquence de la valeur la plus présente
        frequence_max = df[col].value_counts(normalize=True).iloc[0]
        # Si une seule valeur représente plus de 95% de la colonne, on supprime
        if frequence_max > seuil_frequence_texte:
            colonnes_a_supprimer.append(col)
            print(f" {col} : valeur la plus fréquente = {frequence_max:.2f}")

            
    #Suppression effective des colonnes identifiées
    if len(colonnes_a_supprimer) == 0:
        print("Aucune colonne constante trouvée.")
    else:
        for col in colonnes_a_supprimer:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f" {col} supprimée ")
            
    print(f"Shape après suppression : {df.shape}")
    print(f"Colonnes restantes : {list(df.columns)}")
    print("**************")
    return df

# FONCTION 2 : valeurs manquantes
#cherhcher les valurs manquantes et si >50% supprimer la colonne
def supprimer_colonnes_manquantes(df, seuil=50):
    taux_manquant = (df.isnull().sum() / len(df)) * 100
    for col, pourcentage in taux_manquant.items():
        if (pourcentage > 0):
         print(f"{col} : {pourcentage:.2f}% de valeurs manquantes")    # 2. Identifier les colonnes qui dépassent le seuil
    colonnes_a_supprimer = taux_manquant[taux_manquant > seuil].index.tolist()
    # 3. Affichage et suppression
    if not colonnes_a_supprimer:
        print(f" Aucune colonne ne dépasse {seuil}% de valeurs manquantes.")
    else:
        for col in colonnes_a_supprimer:
            pourcentage = taux_manquant[col]
            print(f"{col} : {pourcentage:.2f}% de valeurs manquantes")
        
        # Suppression effective
        df = df.drop(columns=colonnes_a_supprimer)

    print(f"Shape après traitement : {df.shape}")
    print("**************")
    return df


# ============================================================
# FONCTION 2 : Corriger les valeurs aberrantes (AVANT Feature Engineering)
# ============================================================
def corriger_aberrantes(df):
    """
    Corrige les valeurs aberrantes selon les règles métier.
    Transforme les valeurs illogiques en NaN pour un traitement ultérieur.
    """
    print("\n--- CORRECTION DES VALEURS ABERRANTES ---")
    
    # 1. Variables financières et quantités (pas de valeurs négatives)
    colonnes_positives = ['MonetaryTotal', 'MonetaryMin', 'TotalQuantity', 'MinQuantity']
    for col in colonnes_positives:
        if col in df.columns:
            # On compte uniquement les valeurs < 0 (en ignorant les NaN existants)
            nb_neg = df[df[col] < 0].shape[0]
            if nb_neg > 0:
                df.loc[df[col] < 0, col] = np.nan
                print(f" -> {col} : {nb_neg} valeurs négatives → transformées en NaN")

    # 2. Satisfaction Score (doit être entre 0 et 5 selon votre consigne)
    if 'SatisfactionScore' in df.columns:
        # On cherche ce qui n'est PAS entre 0 et 5, tout en ignorant les NaN
        condition_aberrante = ~df['SatisfactionScore'].between(0, 5) & df['SatisfactionScore'].notnull()
        nb = df[condition_aberrante].shape[0]
        if nb > 0:
            df.loc[condition_aberrante, 'SatisfactionScore'] = np.nan
            print(f" -> SatisfactionScore : {nb} valeurs aberrantes → transformées en NaN")

    # 3. Support Tickets (doit être entre 0 et 15)
    if 'SupportTicketsCount' in df.columns:
        condition_aberrante = ~df['SupportTicketsCount'].between(0, 15) & df['SupportTicketsCount'].notnull()
        nb = df[condition_aberrante].shape[0]
        if nb > 0:
            df.loc[condition_aberrante, 'SupportTicketsCount'] = np.nan
            print(f" -> SupportTicketsCount : {nb} valeurs aberrantes → transformées en NaN")

    print(f"Shape après correction : {df.shape}")
    return df


# ============================================================
# FONCTION 3 : Feature Engineering (AVANT suppression des corrélées)
# ============================================================
def feature_engineering(df):
    """
    Création de nouvelles features .
    À faire AVANT la suppression des corrélées car on veut créer 
    toutes les features possibles avant de choisir lesquelles garder.
    """
    print("\n FEATURE ENGINEERING")

    # ── Nouvelles features  ──
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
        print(" MonetaryPerDay = MonetaryTotal / (Recency + 1)")

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
        print(" AvgBasketValue = MonetaryTotal / Frequency")

    # PDF utilise CustomerTenure → votre CSV = CustomerTenureDays
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
        print(" TenureRatio = Recency / CustomerTenureDays")

    # ── Parsing RegistrationDate  ──
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',  # <-- L'ajout MAGIQUE qui supprime l'avertissement
            dayfirst=True,   # priorité format UK
            errors='coerce'  # NaT si format inconnu
        )
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])
        print(" Parsing RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")

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

# FONCTION 4 : Supprimer les colonnes corrolés 
def supprimer_redondantes(df, seuil=0.8, save_reports=True , nom_fichier='correlation_matrix'):
    """
    Pipeline complet :
    1. Calcule et sauvegarde la matrice de corrélation (CSV + image)
    2. Détecte les paires avec |corrélation| > seuil
    3. Supprime la 2ème colonne de chaque paire
    4. Affiche le récapitulatif
    """
    os.makedirs('reports', exist_ok=True)
        
    # 1. Calculer la matrice (valeurs absolues pour la détection)
    df_num = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()    
    corr_abs = corr_matrix.abs()
    
    # 2. Sauvegarder CSV
    if save_reports:
        corr_matrix.to_csv('reports/correlation_matrix.csv')
        print("Matrice CSV sauvegardée : reports/correlation_matrix.csv")
    
    # 3. Sauvegarder heatmap
    if save_reports:
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=False, 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5
        )
        plt.title('Matrice de Corrélation', fontsize=16)
        plt.tight_layout()
        plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(" Heatmap sauvegardée : reports/correlation_matrix.png")
    
    # 4. Détecter les paires corrélées (triangle supérieur pour éviter doublons)
    print(f"\n--- DÉTECTION DES CORRÉLATIONS > {seuil} ---")
    
    mask_triu = np.triu(np.ones_like(corr_abs, dtype=bool), k=1)
    paires = (
        corr_abs.where(mask_triu)
        .stack()
        .reset_index()
    )
    paires.columns = ['Colonne_A', 'Colonne_B', 'Correlation']
    paires = paires[paires['Correlation'] > seuil].sort_values('Correlation', ascending=False)
    
    if paires.empty:
        print("Aucune corrélation forte détectée.")
        return df, []
    
    print(f" {len(paires)} paire(s) trouvée(s) :\n")
    print(paires.to_string(index=False))
    
    # 5. Suppression : on enlève la 2ème colonne (Colonne_B)
    print(f"\n--- SUPPRESSION DES COLONNES REDONDANTES ---")
    colonnes_supprimees = []
    
    for _, row in paires.iterrows():
        col_b = row['Colonne_B']
        if (col_b in df.columns) and(col_b!='Churn') and(col_b!='Recency') :
            df = df.drop(columns=[col_b])
            colonnes_supprimees.append(col_b)
            print(f" {col_b} supprimée (corr = {row['Correlation']:.2f} avec {row['Colonne_A']})")


    print(f"\n{len(colonnes_supprimees)} colonne(s) supprimée(s)")
    print(f" Shape : {corr_matrix.shape[1]} → {df.shape[1]}")
    
    return df, colonnes_supprimees

# FONCTION 6 : Encoder les colonnes texte 
def encoder_colonnes(df):
    """
    Encodage des colonnes texte restantes (PDF pages 3-4).
    - Ordinal  : quand il y a un ordre logique
    - One-Hot  : quand il n'y a pas d'ordre
    - Country  : Target Encoding dans train_model.py après split (PAS ICI)
$    """
    print("\n ENCODAGE DES COLONNES TEXTE")

    # ── Encodage Ordinal  ──
    encodages_ordinaux = {
        'RFMSegment'       : ['Dormants', 'Potentiels', 'Fidèles', 'Champions'],
        'AgeCategory'       : ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'SpendingCategory'  : ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel'      : ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'ChurnRiskCategory'        : ['Faible', 'Moyen', 'Élevé', 'Inconnu'],
        'CustomerType'     : ['Occasionnel', 'Régulier', 'VIP', 'Inconnu'],  # Ordinal selon PDF
    }

    for col, ordre in encodages_ordinaux.items():
        if col in df.columns:
            # Gestion des valeurs inconnues
            df[col] = df[col].apply(lambda x: x if x in ordre else 'Inconnu')
            df[col] = pd.Categorical(df[col], categories=ordre, ordered=True).codes
            print(f" {col} encodé (ordinal)")

    # ── Encodage One-Hot (pas d'ordre → PDF page 3) ──
    colonnes_onehot = [
        'AccountStatus',
        'PreferredTimeOfDay',
        'FavoriteSeason',    # One-Hot  → pas d'ordre entre saisons
        'Region',            # One-Hot  → pas d'ordre entre régions
        'WeekendPreference', # One-Hot  → pas d'ordre
        'ProductDiversity',  # One-Hot  → pas d'ordre
        'Gender',            # One-Hot  → pas d'ordre
    ]

    colonnes_onehot = [c for c in colonnes_onehot if c in df.columns]
    if colonnes_onehot:
        df = pd.get_dummies(df, columns=colonnes_onehot, drop_first=True,dtype=int)
        print(f" One-Hot appliqué sur : {colonnes_onehot}")
        print(df.columns)
    # ── Country → Target Encoding dans train_model.py après split ──
    if 'Country' in df.columns:
        print(" Country gardé → Target Encoding dans train_model.py après split")

    print(df.shape)

    return df


# PIPELINE COMPLET (ORDRE CORRIGÉ)
def pipeline_complet(df):
    """Lance tout le preprocessing dans l'ordre logique"""
    print(" Démarrage du preprocessing...\n")

    # 1. Suppression des colonnes inutiles (constantes uniquement)
    #    CustomerID est gardé pour l'instant (utile pour debug/split)
    df = supprimer_colonnes_inutiles(df)
    # # 2. Correction des aberrantes (avant FE pour avoir des données propres)
    df = corriger_aberrantes(df)
    
    df= supprimer_colonnes_manquantes(df, seuil=50)

    # #    Car on veut créer toutes les features possibles d'abord
    df = feature_engineering(df)
    
    # # 4. Suppression des redondantes 
    # df, _ = supprimer_redondantes(df, seuil=0.8, nom_fichier='correlation_passe1')        
    # # 6. Encodage (sans les features de leakage)
    df = encoder_colonnes(df)
    
    df, _ = supprimer_redondantes(df, seuil=0.8, nom_fichier='correlation_passe1')        


    print(f"\n Preprocessing terminé !")
    print(f"Shape final : {df.shape}")
    print(f"Colonnes finales : {list(df.columns)}")
    print(f"FINI | Shape final : {df.shape}")

    return df


# TEST
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\abdal\Desktop\projects\ML\ML_project\data\raw\retail_customers_COMPLETE CATEGORICAL.csv')
    print(" Données chargées !")

    df_clean = pipeline_complet(df)

    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/data_clean.csv', index=False)