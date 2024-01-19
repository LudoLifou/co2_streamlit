import streamlit as st
import pandas as pd


# --------------------------------Page "Introduction"

def app(df) : 

    st.write("# Analyse des emissions CO2 des voitures")

    # Présentation du projet DS CO2
    st.write("""
             #### Objectifs:  
             - Construire un modèle d'apprentissage automatique fiable pour prédire les émissions de CO2 dans différents types de voitures, en fonction de leurs caractéristiques:  
                - Puissance  
                - Masse  
                - Type de carburant  
                - etc...  
             - Fournir des outils d'analyse sur l'influence de ces caractéristiques, et sur leur répartition dans le marché automobile  
             - Développer des connaissances dans ce domaine pour accompagner les décideurs privés et publics

             le Transport est responsables de 25% des émissions de CO2 dans le monde
             """)
             

   
    ## Enjeux du sujet


    ## Base de données

    st.write(
        "#### DataSet:", "\n",
        "Source du jeu de données: European Environment Agency (EEA)  [European Environment Agency](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b)  ", "\n",

        "C'est une structure de l'Union Europeenne qui enregistre toutes les nouvelles immatriculations de voitures en europe.  ", "\n",
        "La taille du ce jeu de données est très importante, avec par exemple 10 millions de véhicules uniquement pour l'année 2021."
    )

    st.write("""
             #### Variable cible: 
             **=> Emission de CO2 des voitures**, exprimée en g/km  

             C'est un problème de regression, mais pour des raisons pédagogiques, nous avons souhaité explorer deux approches :  
            - **Problème de Regression**: On cherche à prédire la valeur des émisssions de CO2. 
             - **Problème de Classification**: On cherche à prédire la classe d'apartenance des émisssions de CO2.  

             En effet, le dfshfsdhf définit un classement pour le niveau de CO2 emis, et on peut imaginer que l'objectif soit uniquement de prédire ce classement.  
             """)

    st.write("")
    st.write("### Description")
    st.write("")
    st.write("")



    
    ### ---------------------------------Les données sources------------------------------------------
    # TODO: reordonner le df 
    st.write("##### Les données")
    """
    
    """

    # Affichage optionnel 
    # Reordonner les colonnes pour voir la différence preprocess reg, class + selectionner certaines colonnes.
    # Utiliser le code suivant  
    if st.checkbox('Avant preprocessings'):     # chekbox optionel
        st.dataframe(df.sample(15))
    else :
        st.dataframe(df.sample(15))    
        if st.checkbox('choisir les variables'):
            options = st.multiselect(
            'Colonnes à afficher',
            df.columns)


    if st.checkbox('Voir le résumé'):     # chekbox optionel
        st.dataframe(df.describe())

    st.write("##### Queqlues chiffres")
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

 
    if st.checkbox('voir quelques variables'):
        options = st.multiselect(
        'Colonnes à afficher',
        df.columns)
        

        # st.write(options)
        st.write(df[options].sample(n = 20))

# Page "DataViz"
# Il faudra importer un df propre avec plus de variables ?

# Idée de graph interactif = corrélation CO2/Autre variable (distplot.. autres)