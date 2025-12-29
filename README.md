# SYSTÈME DE PRÉDICTION DE CHURN CLIENT

**Description :**
- Projet de classification pour prédire le churn (désabonnement) des clients d'une entreprise télécom. Comprend l'exploration des données, l'entraînement d'un modèle, et une interface Streamlit pour tester des profils.

**Objectifs :**
- Fournir un modèle capable d'identifier les clients à risque de churn.
- Exposer une interface utilisateur pour évaluer des cas réels.

**Résultats (compléter après entraînement)**
- **Précision globale :**
- **Recall (classe churn) :**
- **F1-score (classe churn) :**
- **Taille du jeu de données :**

**Stack technique :**
- **Langage :** Python
- **Librairies :** scikit-learn, pandas, numpy, streamlit
- **Données :** Telco Customer Churn (CSV)

**Arborescence du projet**

```
CHURN CLIENT/
	app.py
	train_model.py
	README.md
	requirements.txt
	telco_churn.csv
	models/
	notebooks/
		01_eda_and_training.ipynb
	screens/
```

**Fichiers principaux :**
- **app.py** : Interface Streamlit pour tester des profils et visualiser des métriques. Voir [app.py](app.py).
- **train_model.py** : Script d'entraînement et de sauvegarde du modèle. Voir [train_model.py](train_model.py).
- **telco_churn.csv** : Jeu de données brut. Voir [telco_churn.csv](telco_churn.csv).
- **models/** : Dossier contenant les modèles entraînés et artefacts.
- **notebooks/01_eda_and_training.ipynb** : Notebook d'EDA et d'entraînement.

**Installation**

1. Créez un environnement virtuel (recommandé) :

```bash
python -m venv .venv
source .venv/bin/activate   # Unix/macOS
.venv\\Scripts\\Activate    # Windows PowerShell
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

3. Entraînez le modèle (optionnel si un modèle est déjà présent dans `models/`) :

```bash
python train_model.py
```

4. Lancez l'application Streamlit :

```bash
python -m streamlit run app.py
```

**Utilisation**
- Ouvrez l'URL fournie par Streamlit (généralement `http://localhost:8501`).
- Utilisez les contrôles pour saisir les caractéristiques d'un client et obtenir la probabilité de churn et la prédiction.

**Détails techniques**
- Prétraitement : gestion des valeurs manquantes, encodage des variables catégorielles, normalisation si nécessaire.
- Modèle : classification binaire (ex. RandomForest / LogisticRegression). Les métriques importantes sont précision, rappel, F1-score et courbe ROC-AUC.
- Artefacts : le modèle entraîné et les objets de pipeline (vectoriseur / scaler) sont sauvegardés dans `models/`.

**Exemples de commandes rapides**

```bash
# Entraîner et sauvegarder le modèle
python train_model.py

# Lancer l'interface utilisateur
python -m streamlit run app.py
```