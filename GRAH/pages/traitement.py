import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier




st.write("installez les packages necessaires et importez vos donnees et effectuez la phase d'exploration de base des donnees")

df = pd.read_csv("Expresso_chun_dataset.csv")

st.write(df.head())

st.write("Afficher des informations generales sur l'ensemble de donnees")

st.write(df.info())
st.write("l'information générale se trouve au niveau du terminal")

# Créer des rapports de profilage de pandas pour obtenir des informations sur l'ensemble de données
st.write("Rapport de profilage de l'ensemble de données :")
st.write(df.describe())

# Gérer les valeurs manquantes et corrompues
missing_values = df.isnull().sum()
st.write("Valeurs manquantes dans les données :")
st.write(missing_values)

for col in df.columns:
    if df[col].dtype == 'object':
        # Si la colonne est catégorielle, remplacez les valeurs manquantes par le mode
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

    elif df[col].dtype in ['int64','float64']:
        # Sinon, si c'est numérique, remplacez les valeurs manquantes par 0
        mean_val = 0
        df[col].fillna(mean_val, inplace=True)

    else:
        pass
st.write("Vérifions si elles ont bien été géré :")

missing_values = df.isnull().sum()
st.write(missing_values)

# Supprimer les doublons, s'ils existent
df.drop_duplicates(inplace=True)
st.write("Doublons supprimés. Nouvelle taille de l'ensemble de données :", df.shape)

st.write("Gérer les valeurs aberrantes, si elles existent")
# Fonction pour détecter les valeurs aberrantes basées sur le z-score
def detect_outliers_zscore(data, threshold=3):
    outliers = []
    z_scores = (df - df.mean()) / data.std()
    outlier_indices = pd.abs(z_scores) > threshold
    outliers = data[outlier_indices]
    return outliers
st.write(df.head(100))

st.write("SELECTION DES COLONNES CATEGORIELLES")

colonnes_categoriell = df.select_dtypes(include=['object']).columns

# Initialiser LabelEncoder
st.write("ENCODAGE")

label_encoder = LabelEncoder()

# Copie des données pour éviter les modifications sur les données originales
#encoded_data = df.copy()

label_encoder = LabelEncoder()
for colonne in colonnes_categoriell:
    df[colonne] = label_encoder.fit_transform(df[colonne])

#df['CHURN'] = label_encoder.fit_transform(df["CHURN"])
# Affichage des données encodées
print("Données encodées :")
st.write(df.head(100))

st.write("3.base sur l'entrainement d'exploration de donnees precedent et test d'un classificateur d'apprentissage automatique")

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

st.write("Accuracy Score:", accuracy)





