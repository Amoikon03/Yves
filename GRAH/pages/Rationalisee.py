import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.write('4.creer une application rationalisee (localement)')


# df = pd.read_csv(GRAH/Expresso_chun_dataset.csv")

# st.write(df.head(100))


def plot_sin_wave(x_min, x_max, num_points, frequency):
    x = np.linspace(x_min, x_max, num_points)
    y = np.sin(frequency * x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Sin(X)')
    ax.set_title('Courbe Sinus')

    return fig


def app():
    st.title("Tracer une courbe sinus")

    # Ajout des champs de saisie pour les fonctionnalités
    x_min = st.number_input("Valeur minimale de X", value=0.0)
    x_max = st.number_input("Valeur maximale de X", value=10.0)
    num_points = st.number_input("Nombre de points", value=100)
    frequency = st.number_input("Fréquence de la sinusoïde", value=1.0)

    # Bouton de validation
    if st.button("Valider"):
        # Tracer la courbe sinus avec les paramètres saisis
        fig = plot_sin_wave(x_min, x_max, num_points, frequency)
        st.pyplot(fig)


# Lancement de l'application
app()
