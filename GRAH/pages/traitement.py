import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Affichez un message initial pour l'utilisateur
st.write("Installez les packages nécessaires, importez vos données et effectuez la phase d'exploration de base des données")

# Charger les données depuis un fichier CSV
df = pd.read_csv("GRAH/Expresso_chun_dataset.csv")

# Afficher les premières lignes du DataFrame
st.write(df.head())

# Afficher des informations générales sur l'ensemble de données
st.write("Afficher des informations générales sur l'ensemble de données")
st.write(df.info())  # Ceci affiche des informations au niveau du terminal
st.write("L'information générale se trouve au niveau du terminal")

# Créer des rapports de profilage de pandas pour obtenir des informations sur l'ensemble de données
st.write("Rapport de profilage de l'ensemble de données :")
st.write(df.describe())

# Gérer les valeurs manquantes et corrompues
missing_values = df.isnull().sum()
st.write("Valeurs manquantes dans les données :")
st.write(missing_values)

# Remplacer les valeurs manquantes par le mode ou la moyenne selon le type de données
for col in df.columns:
    if df[col].dtype == 'object':
        # Si la colonne est catégorielle, remplacez les valeurs manquantes par le mode
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    elif df[col].dtype in ['int64','float64']:
        # Sinon, si c'est numérique, remplacez les valeurs manquantes par 0
        mean_val = 0
        df[col].fillna(mean_val, inplace=True)

# Vérifier si les valeurs manquantes ont été bien gérées
st.write("Vérifions si elles ont bien été gérées :")
missing_values = df.isnull().sum()
st.write(missing_values)

# Supprimer les doublons, s'ils existent
df.drop_duplicates(inplace=True)
st.write("Doublons supprimés. Nouvelle taille de l'ensemble de données :", df.shape)

# Gérer les valeurs aberrantes, si elles existent
st.write("Gérer les valeurs aberrantes, si elles existent")
def detect_outliers_zscore(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    outlier_indices = (z_scores.abs() > threshold).any(axis=1)
    return data[outlier_indices]

# Afficher les premières lignes du DataFrame après le traitement des valeurs aberrantes
st.write(df.head(100))

# Sélection des colonnes catégorielles
st.write("Sélection des colonnes catégorielles")
colonnes_categorielles = df.select_dtypes(include=['object']).columns

# Encodage des colonnes catégorielles avec LabelEncoder
st.write("Encodage des colonnes catégorielles")
label_encoder = LabelEncoder()
for colonne in colonnes_categorielles:
    df[colonne] = label_encoder.fit_transform(df[colonne])

# Affichage des données encodées
st.write("Données encodées :")
st.write(df.head(100))

# Préparation pour l'entraînement du modèle de classification
st.write("Préparation pour l'entraînement et le test d'un classificateur d'apprentissage automatique")

# Séparer les caractéristiques (features) et la cible (target)
X = df.drop(['CHURN'], axis=1)
y = df['CHURN']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le classificateur RandomForestClassifier
classifier = RandomForestClassifier()

# Entraîner le classificateur sur l'ensemble d'entraînement
classifier.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = classifier.predict(X_test)

# Calculer l'exactitude des prédictions
accuracy = accuracy_score(y_test, y_pred)
st.write("Score de précision (Accuracy Score) :", accuracy)
