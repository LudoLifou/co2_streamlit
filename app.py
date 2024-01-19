# Le streamlit devra eter suport de présentation ? Voir idées des autres

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import intro
import dataviz
import model
import demo
import conclusion
import exemple
import accueil

# Voir si c'est a mettre dans les pages ou la.
st.set_page_config(layout="wide")   # pb eventuel 
st.set_option('deprecation.showPyplotGlobalUse', False)

#----------------------------------------------Chargement des données

# Session State provides a dictionary-like interface where you can save information that is preserved between script reruns or multipage application
# st.session_state["my_key"] or st.session_state.my_key.

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_pickle('./data/c02_reduit_2e5.pkl')
    return df
# TODO: import un fichier plus petit

# Chargement des données    
if 'df' not in st.session_state:            # ou df = load_data()            
    st.session_state['df'] = load_data()
df = st.session_state.df

#------------------------------------------------SOMMAIRE 

st.sidebar.title("Emissions de CO2 des vehicules")
PAGES = { 
    "Acceuil": accueil,
    "Présentation": intro,
    "Explorations": dataviz,
    "Modélisation": model,
    "Démonstration": demo,
    "Conclusion et perspectives": conclusion,
    "Exemples et idees streamlit": exemple
    }

selection = st.sidebar.radio("Menu", list(PAGES.keys()))
page = PAGES[selection]
page.app(df)

st.sidebar.image(
        "./data/images/pollution3.png",
        width=250,
    )

## Affichage des auteurs et mentor en bas de la sidebar:
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write('### Auteurs:')
st.sidebar.write('Ludovic Calmettes')
st.sidebar.write('Ludovic Calmettes')
st.sidebar.write('Ludovic Calmettes')
st.sidebar.write('Ludovic Calmettes')
st.sidebar.write(' ')
st.sidebar.write('### Mentor:')
st.sidebar.write('Khalil')

