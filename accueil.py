import streamlit as st
from PIL import Image


def app(df):

    background = Image.open("./data/images/voiture_ciel.png")
    st.image(background,
               use_container_width=True)

