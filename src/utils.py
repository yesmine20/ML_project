#  les outils génériques (afficher des stats, détecter les valeurs manquantes...)
#fonction shape de données 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# FONCTION 1 : Shape des données

def analyser_shape(df):
    """Affiche le nombre de clients et de features"""
    print("=" * 50)
    print("📊 DIMENSIONS DU DATASET")
    print("=" * 50)
    lignes, colonnes = df.shape
    print(f"Nombre de clients   : {lignes}")
    print(f"Nombre de features  : {colonnes}")


# ============================================================
# FONCTION 2 : Valeurs manquantes
# ============================================================
def analyser_manquants(df):
    """Détecte et affiche les colonnes avec valeurs manquantes"""
    print("\n" + "=" * 50)
    print(" VALEURS MANQUANTES")
    print("=" * 50)
    manquants = df.isnull().sum()
    manquants = manquants[manquants > 0]
    
    if manquants.empty:
        print(" Aucune valeur manquante !")
        return []
    else:
        pct = (manquants / len(df) * 100).round(2)
        resultat = pd.DataFrame({
            'Nombre manquants': manquants,
            'Pourcentage (%)': pct
        })
        print(resultat)
        return manquants.index.tolist()


# FONCTION 3 : Valeurs aberrantes

def analyser_aberrantes(df):
    """Détecte les valeurs aberrantes avec describe() et boxplots"""
    print("\n" + "=" * 50)
    print("  STATISTIQUES (chercher min/max suspects)")
    print("=" * 50)
    print(df.describe().round(2))
    
    # Boxplots pour les colonnes numériques
    numeriques = df.select_dtypes(include='number').columns.tolist()
    # On exclut CustomerID car c'est juste un identifiant
    numeriques = [c for c in numeriques if c != 'CustomerID']
    
    # print(f"\n Boxplots pour détecter les outliers...")
    # fig, axes = plt.subplots(
    #     nrows=(len(numeriques) // 4) + 1,
    #     ncols=4,
    #     figsize=(20, 40)
    # )
    # axes = axes.flatten()
    
    # for i, col in enumerate(numeriques):
    #     df.boxplot(column=col, ax=axes[i])
    #     axes[i].set_title(col, fontsize=8)
    
    # plt.tight_layout()
    # plt.savefig('reports/boxplots_outliers.png')
    # print("Boxplots sauvegardés dans reports/boxplots_outliers.png")
    # plt.show()
    return numeriques 



# FONCTION 4 : Types des colonnes

def analyser_types(df):
    """Identifie les colonnes texte et numériques"""
    print("\n" + "=" * 50)
    print("TYPES DES COLONNES")
    print("=" * 50)
    
    numeriques = df.select_dtypes(include='number').columns.tolist()
    texte = df.select_dtypes(include='object').columns.tolist()
    
    print(f"\n Colonnes numériques ({len(numeriques)}) :")
    print(numeriques)
    print(f"\n Colonnes texte à encoder ({len(texte)}) :")
    print(texte)
    return numeriques, texte



# FONCTION 5 : Colonnes inutiles

def analyser_uniques(df):
    """Détecte les colonnes avec une seule valeur (inutiles)"""
    print("\n" + "=" * 50)
    print(" COLONNES POTENTIELLEMENT INUTILES")
    print("=" * 50)
    # calcule le nombre de valeurs uniques pour chaque colonne
    uniques = df.nunique()
    
    # Colonnes avec 1 seule valeur = inutiles
    # filtre pour garder uniquement les colonnes inutiles (constantes)
    inutiles = uniques[uniques == 1]
    if not inutiles.empty:
        print(f" Colonnes avec 1 seule valeur (à supprimer) :")
        print(inutiles)
    else:
        print(" Pas de colonne complètement inutile")
    
    # Colonnes avec trop de valeurs uniques (ex: IP, ID)
    print(f"\n  Colonnes avec beaucoup de valeurs uniques :")
    suspects = uniques[uniques > len(df) * 0.9]
    print(suspects)
    return inutiles.index.tolist() 

# FONCTION 6 : Échelles

def analyser_echelles(df):
    """Affiche les min/max pour voir les différences d'échelles"""
    print("\n" + "=" * 50)
    print("📏 ÉCHELLES DES COLONNES NUMÉRIQUES")
    print("=" * 50)
    
    numeriques = df.select_dtypes(include='number')
    echelles = pd.DataFrame({
        'Min': numeriques.min(),
        'Max': numeriques.max(),
        'Écart': numeriques.max() - numeriques.min()
    }).sort_values('Écart', ascending=False)
    
    print(echelles.round(2))
    print("\n Les colonnes avec un grand écart nécessitent une normalisation !")
    colonnes_a_normaliser = echelles[echelles['Écart'] > 1].index.tolist()
    return colonnes_a_normaliser



# FONCTION 7 : Distribution de la cible (Churn)

def analyser_cible(df, cible='Churn'):
    """Vérifie si les classes sont équilibrées"""
    print("\n" + "=" * 50)
    print(f"DISTRIBUTION DE LA CIBLE : {cible}")
    print("=" * 50)
    
    counts = df[cible].value_counts()
    pct = (counts / len(df) * 100).round(2)
    
    print(pd.DataFrame({'Nombre': counts, 'Pourcentage (%)': pct}))
    
    if pct.min() < 20:
        print("\n  Classes déséquilibrées ! Penser au rééquilibrage.")
    else:
        print("\n Classes relativement équilibrées.")



# FONCTION  : Colonnes redondantes

def analyser_redondantes(df, seuil=0.8):
    """Détecte, affiche et retourne les colonnes redondantes"""
    print("\n" + "=" * 50)
    print(f" COLONNES REDONDANTES (corrélation > {seuil})")
    print("=" * 50)

    correlation = df.corr(numeric_only=True)
    colonnes_a_supprimer = []

    for col in correlation.columns:
        for row in correlation.index:
            if col != row and col != 'Churn' and row != 'Churn':
                if abs(correlation[col][row]) > seuil:
                    # On garde col, on supprime row
                    if row not in colonnes_a_supprimer and col not in colonnes_a_supprimer:
                        colonnes_a_supprimer.append(row)
                        print(f" {row} redondante avec {col} "
                              f"(corrélation = {correlation[col][row]:.2f})")

    print(f"\n→ {len(colonnes_a_supprimer)} colonnes redondantes détectées")
    return colonnes_a_supprimer 

# RAPPORT COMPLET : appelle toutes les fonctions

def rapport_complet(df):
    analyser_shape(df)
    analyser_manquants(df)
    analyser_types(df)
    analyser_uniques(df)
    analyser_redondantes(df)
    analyser_echelles(df)
    analyser_cible(df)
    analyser_aberrantes(df)

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\abdal\Desktop\projects\ML\ML_project\data\raw\retail_customers_COMPLETE CATEGORICAL.csv')
    print("Données chargées !")
    rapport_complet(df)
    