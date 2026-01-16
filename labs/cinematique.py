import streamlit as st
import pandas as pd
from utils.supabase_client import supabase

def run_cinematique_lab():
    st.header("Laboratoire Cinématique 1D - Données brutes")

    st.write("Entrez vos mesures de position x (m) et temps t (s).")

    # Créer des listes vides pour stocker les entrées
    x_list = []
    t_list = []

    # Nombre de mesures à entrer
    n = st.number_input("Nombre de mesures", min_value=2, max_value=50, value=5, step=1)

    # Entrée des mesures
    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"x[{i}] (m)", value=0.0, key=f"x{i}")
        with col2:
            t_val = st.number_input(f"t[{i}] (s)", value=0.0, key=f"t{i}")
        x_list.append(x_val)
        t_list.append(t_val)

    # Bouton pour envoyer les données
    if st.button("Envoyer les données brutes"):
        # Préparer le JSON à stocker
        data_brute = {
            "x": x_list,
            "t": t_list
        }

        # Envoyer sur Supabase
        supabase.table("cinematique_brute").insert({"results": data_brute}).execute()
        st.success("Données brutes envoyées sur Supabase ! ✅")

        # Afficher les données
        df = pd.DataFrame(data_brute)
        st.dataframe(df)
        st.line_chart(df)
