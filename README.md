# Analyse Comportementale Clientèle Retail
> Pipeline ML complet : Clustering RFM · Classification Churn · Régression MonetaryTotal

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green) ![Flask](https://img.shields.io/badge/Flask-API-lightgrey)

---

## Description

Solution d'analyse comportementale pour une plateforme e-commerce de cadeaux. À partir de **4 372 clients** et **52 variables**, trois modèles complémentaires ont été développés et déployés via une API Flask :

| Objectif | Algorithme | Résultat |
|---|---|---|
| Segmentation clients | K-Means (k=4) | 4 segments RFM identifiés |
| Prédiction du churn | Random Forest | AUC-ROC : **97.6%** · Accuracy : **92.2%** |
| Estimation des dépenses | XGBoost | MAPE : **0.9%** · R² : **0.950** |

---

## Structure du Projet

```
├── data/
│   └── retail_customers.csv       # Dataset brut
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Nettoyage & feature engineering
│   ├── 02_clustering.ipynb        # Segmentation K-Means RFM
│   ├── 03_classification.ipynb    # Prédiction churn Random Forest
│   └── 04_regression.ipynb        # Estimation dépenses XGBoost
├── models/
│   ├── kmeans_model.pkl
│   ├── rf_churn_model.pkl
│   ├── xgb_monetary_model.pkl
│   └── pipeline_preprocessor.pkl  # Imputer + scaler + encodeur
├── app/
│   ├── app.py                     # API Flask
│   └── templates/
│       └── index.html             # Interface utilisateur
└── requirements.txt
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-repo/ml-retail-analysis.git
cd ml-retail-analysis

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
flask
joblib
matplotlib
seaborn
```

---

## Usage

### 1. Entraîner les modèles
Exécuter les notebooks dans l'ordre :
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```
Les modèles `.pkl` sont sauvegardés automatiquement dans `models/`.

### 2. Lancer l'API Flask
```bash
cd app
python app.py
# → http://localhost:5000
```

### 3. Faire une prédiction (API REST)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 45,
    "Frequency": 8,
    "MonetaryTotal": 1200,
    "Age": 34,
    "SatisfactionScore": 4
  }'
```

**Réponse JSON :**
```json
{
  "churn_probability": 0.23,
  "churn_risk": "Faible",
  "rfm_segment": "Champions",
  "estimated_spend": 1340.5,
  "recommendation": "Programme VIP, early access nouveautés"
}
```

---

## Résultats ML

### Classification Churn — Random Forest

| Métrique | Valeur |
|---|---|
| Accuracy | 92.2% |
| AUC-ROC | 97.6% |
| F1-Score | 88.03% |
| Précision | 90.3% |
| Rappel | 85.9% |

**Matrice de confusion (875 clients test) :**
```
                 Prédit Fidèle   Prédit Churn
Réel Fidèle          557             27
Réel Churn            41            250
```

**Top 5 features :**
1. `PreferredMonth` — 16.99%
2. `FirstPurchaseDaysAgo` — 13.20%
3. `Frequency` — 9.62%
4. `AvgDaysBetweenPurchases` — 6.13%
5. `SpendingCategory` — 6.09%

### Régression MonetaryTotal — Comparaison

| Modèle | MAPE | R² | MAE |
|---|---|---|---|
| **XGBoost** ✅ | **0.9%** | **0.950** | **141 €** |
| LightGBM | 2.0% | 0.781 | 249 € |
| Random Forest | 7.2% | 0.412 | 493 € |

---

## Pipeline de Preprocessing

```
Données brutes
    ↓ Suppression colonnes variance nulle
    ↓ Correction valeurs aberrantes (règles métier)
    ↓ Suppression colonnes > 50% manquant
    ↓ Feature Engineering (MonetaryPerDay, AvgBasketValue, TenureRatio, PurchaseIntensity)
    ↓ Encodage (Ordinal / One-Hot / Target Encoding)
    ↓ Suppression colonnes corrélées (> 0.8)
    ↓ SMOTE (train uniquement, classification)
Données prêtes
```

---

## Déploiement Flask

L'API expose un endpoint unique `/predict` qui :
1. Reconstruit les features dérivées
2. Applique le pipeline de transformation sauvegardé
3. Interroge les 3 modèles en séquence
4. Retourne segment RFM + probabilité churn + estimation dépenses + recommandation marketing

> **Important :** Le pipeline `pipeline_preprocessor.pkl` doit être chargé avec les modèles pour garantir des transformations identiques à l'entraînement.

---

## Encadrante

**Mme Fadoua Drira** — Module Machine Learning, GI2  
École Supérieure d'Ingénierie · 2025–2026
