import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Configuration du design
st.set_page_config(page_title="Scanner de Ventes", layout="wide")

st.title("📊 Tableau de Bord : Prédiction des Ventes")
st.markdown("Outil d'aide à la décision basé sur l'Intelligence Artificielle.")
st.divider()

# 2. L'Interaction Client : Le glisser-déposer
fichier_client = st.file_uploader("📁 Importez votre historique de ventes (Format CSV)", type=['csv'])

# Le cerveau ne s'active que SI un fichier est importé
if fichier_client is not None:
    
    # Lecture des données du client
    df = pd.read_csv(fichier_client)
    
    # Sécurité : On vérifie que le client a bien nommé ses colonnes 'mois' et 'ventes'
    if 'mois' in df.columns and 'ventes' in df.columns:
        
        # --- LE MOTEUR MATHÉMATIQUE ---
        X = df[['mois']].values 
        y = df['ventes'].values

        model = LinearRegression()
        model.fit(X, y)

        # On détecte automatiquement quel est le dernier mois pour prédire le suivant
        dernier_mois = int(df['mois'].max())
        mois_futur = np.array([[dernier_mois + 1]])
        
        prediction = model.predict(mois_futur)
        fiabilite = model.score(X, y)

        # --- L'INTERFACE VISUELLE ---
        st.success("Analyse terminée avec succès !")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"🔮 Prévision pour le mois {dernier_mois + 1}", value=f"{prediction[0]:,.2f} DH")
        with col2:
            st.metric(label="🎯 Indice de Confiance (R²)", value=f"{fiabilite*100:.1f} %")

        st.subheader("Analyse Visuelle de la Tendance")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(X, y, color='blue', label='Ventes réelles (Passé)')
        ax.plot(X, model.predict(X), color='red', linestyle='--', label='Tendance calculée')
        ax.scatter(mois_futur, prediction, color='green', marker='*', s=200, label='Prédiction')
        ax.set_xlabel('Mois')
        ax.set_ylabel('Chiffre d\'Affaires (DH)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Données brutes du client")
        st.dataframe(df.T)
        
    else:
        st.error("⚠️ Erreur : Le fichier doit contenir les colonnes 'mois' et 'ventes' exactes.")
else:
    # Ce qui s'affiche quand il n'y a pas encore de fichier
    st.info("👋 Bienvenue ! En attente d'un fichier de données pour lancer l'intelligence artificielle.")