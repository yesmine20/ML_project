# ML_project
🧠 Analyse Comportementale Clientèle – Retail E-Commerce
📌 Contexte

Projet réalisé dans le cadre du module Machine Learning – GI2 (2025-2026).

Nous jouons le rôle de Data Scientist dans une entreprise e-commerce spécialisée dans les cadeaux.
L’objectif est d’analyser le comportement des clients afin de :

🎯 Personnaliser les stratégies marketing

📉 Réduire le churn (départ des clients)

💰 Optimiser le chiffre d’affaires

📊 Segmenter intelligemment la clientèle

Le dataset contient 52 features (numériques et catégorielles) issues de transactions réelles et comporte volontairement des problèmes de qualité.

🏗️ Structure du Projet
projet_ml_retail/
│
├── data/
│   ├── raw/              # Données brutes
│   ├── processed/        # Données nettoyées
│   └── train_test/       # Données splitées
│
├── notebooks/            # Prototypage Jupyter
│
├── src/
│   ├── preprocessing.py  # Nettoyage & feature engineering
│   ├── train_model.py    # Entraînement modèles
│   ├── predict.py        # Prédictions
│   └── utils.py          # Fonctions utilitaires
│
├── models/               # Modèles sauvegardés (.pkl / .joblib)
├── app/                  # Application Flask
├── reports/              # Visualisations & résultats
│
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation
1️⃣ Cloner le projet
git clone <votre_lien_github>
cd projet_ml_retail
2️⃣ Créer l’environnement virtuel
python -m venv venv
Activation :

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3️⃣ Installer les dépendances
pip install -r requirements.txt
📊 Pipeline Machine Learning

Le projet suit la chaîne complète Data Science :

1️⃣ Exploration des données

Analyse descriptive

Détection des valeurs manquantes

Analyse des corrélations (heatmap)

Détection de multicolinéarité (VIF)

2️⃣ Préparation & Nettoyage

Imputation des valeurs manquantes (Mean / Median / KNN)

Parsing des dates (RegistrationDate)

Suppression des features inutiles

Encodage des variables catégorielles

Normalisation (StandardScaler)

Gestion du déséquilibre des classes

⚠️ Le scaler est appliqué uniquement sur X_train pour éviter le data leakage.