import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Analyse des Données Alimentaires", layout="wide")

nutriscore_palette = {
    'A': '#008000',  # Vert
    'B': '#85BB2F',  # Vert clair
    'C': '#FFD700',  # Jaune
    'D': '#FF8000',  # Orange
    'E': '#FF0000'   # Rouge
}

# Chargement des données
@st.cache
def load_data():
    return pd.read_csv('dataset_cleaned_fr.csv')

df = load_data()

# Titre principal
st.title("Exploration et Analyse des Données Alimentaires")

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Choisissez une section :", 
    ["Aperçu des données", "Analyse Univariée", "Analyse Bivariée", "Analyse Multivariée"]
)

# Section : Aperçu des données
if options == "Aperçu des données":
    st.header("Aperçu des données")
    st.write("Voici un aperçu des premières lignes du dataset :")
    st.dataframe(df.head())
    st.write("Statistiques descriptives :")
    st.write(df.describe())

# Section : Analyse Univariée
elif options == "Analyse Univariée":
    st.header("Analyse Univariée")
    st.write("Distribution des Nutri-Scores :")
    nutriscore_palette = {'A': '#008000', 'B': '#85BB2F', 'C': '#FFD700', 'D': '#FF8000', 'E': '#FF0000'}
    nutriscore_counts = df['nutrition_grade_fr'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        nutriscore_counts, 
        labels=nutriscore_counts.index, 
        autopct='%1.1f%%', 
        colors=[nutriscore_palette[label] for label in nutriscore_counts.index], 
        startangle=90
    )
    ax.set_title("Répartition des Nutri-Scores")
    st.pyplot(fig)

    st.write("Distribution des catégories PNNS :")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df, 
        y='pnns_groups_combined', 
        order=df['pnns_groups_combined'].value_counts().index, 
        palette='viridis', 
        ax=ax
    )
    ax.set_title("Nombre de produits par catégorie PNNS")
    st.pyplot(fig)


# Section : Analyse Bivariée
elif options == "Analyse Bivariée":
    st.header("Analyse Bivariée")
    st.write("Corrélation entre les variables numériques :")
    numerical_cols = ['sugars_100g', 'fat_100g', 'carbohydrates_100g', 'energy_100g', 'nutrition-score-fr_100g']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)

    st.write("Répartition du Nutri-Score par catégorie PNNS :")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.countplot(
        data=df, 
        x='pnns_groups_combined', 
        hue='nutrition_grade_fr', 
        palette=nutriscore_palette,  # Correction ici
        ax=ax
    )
    ax.set_title("Répartition du Nutri-Score par catégorie PNNS")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

# Section : Analyse Multivariée
elif options == "Analyse Multivariée":
    st.header("Analyse Multivariée")
    st.write("Analyse en Composantes Principales (ACP) :")

    # Préparation des données pour l'ACP
    features = ['sugars_100g', 'fat_100g', 'carbohydrates_100g', 'energy_100g', 'nutrition-score-fr_100g']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de l'ACP
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['F1', 'F2'])
    df_pca['pnns_groups_combined'] = df.loc[X.index, 'pnns_groups_combined']

    # Graphique 1 : Projection des individus
    st.subheader("Projection des individus (F1 vs F2)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_pca, 
        x='F1', 
        y='F2', 
        hue='pnns_groups_combined', 
        palette='tab10', 
        ax=ax1, 
        alpha=0.8
    )
    ax1.set_title("Projection des individus (F1 vs F2)")
    st.pyplot(fig1)

    # Graphique 2 : Cercle des corrélations
    st.subheader("Cercle des corrélations (F1 vs F2)")
    pca_components = pca.components_
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    for i, var in enumerate(features):
        ax2.arrow(0, 0, pca_components[0, i], pca_components[1, i], 
                  head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax2.text(pca_components[0, i] + 0.02, pca_components[1, i] + 0.02, var, fontsize=12)

    ax2.plot([-1, 1], [0, 0], color='grey', ls='--')
    ax2.plot([0, 0], [-1, 1], color='grey', ls='--')
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b', linestyle='--')
    ax2.add_patch(circle)

    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel('F1')
    ax2.set_ylabel('F2')
    ax2.set_title('Cercle des corrélations (F1 vs F2)')
    ax2.grid()
    st.pyplot(fig2)