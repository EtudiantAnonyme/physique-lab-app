import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase
import io

def run_cinematique_2D_lab():
    st.set_page_config(
        page_title="Laboratoire Cinématique 2D",
        page_icon="🧭",
        layout="wide"
    )
 
    st.subheader("🧭 Laboratoire Cinématique 2D — Cégep Montmorency")
    st.markdown("""
    Cette application permet de :
    - Enregistrer et gérer des mesures expérimentales en 2D
    - Ajuster des modèles linéaires et quadratiques en 2D
    - Visualiser graphiquement les résultats en 2D
    - Calculer vitesse et accélération en 2D
    - Tester différents temps ou valeurs pour comprendre les phénomènes cinématiques en 2D
    """)

    st.divider()
