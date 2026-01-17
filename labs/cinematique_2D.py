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
    # =======================
    # 1️⃣ Type d’expérience
    # =======================
    st.header("1️⃣ Type d’expérience")

    exp_type = st.selectbox(
        "Choisissez le type d’expérience",
        [
            "Projectile / Catapulte",
            "Mouvement plan général",
            "Mouvement circulaire (à venir)"
        ]
    )
    st.divider()
    # =======================
    # 2️⃣ Ajouter des mesures expérimentales
    # =======================
    st.header("2️⃣ Ajouter des données expérimentales")

    n = st.number_input(
    "Nombre de mesures",
    min_value=2,
    max_value=100,
    value=10,
    step=1
    )   
    if exp_type == "Projectile / Catapulte":
        st.markdown("### 🎯 Données — Mouvement balistique")

        t_list, x_list, y_list = [], [], []

        for i in range(n):
            c1, c2, c3 = st.columns(3)
            with c1:
                t = st.number_input(f"t[{i}] (s)", key=f"t_cat_{i}")
            with c2:
                x = st.number_input(f"x[{i}] (m)", key=f"x_cat_{i}")
            with c3:
                y = st.number_input(f"y[{i}] (m)", key=f"y_cat_{i}")

            t_list.append(t)
            x_list.append(x)
            y_list.append(y)

        angle_known = st.checkbox("Angle de lancement connu ?")

        theta = None
        if angle_known:
            theta = st.number_input("Angle θ (degrés)", value=45.0)

        if st.button("📤 Enregistrer l’expérience"):
            supabase.table("cinematique_2D").insert({
                "type": "catapulte",
                "results": {
                    "t": t_list,
                    "x": x_list,
                    "y": y_list,
                    "theta": theta
                }
            }).execute()

            st.success("✅ Données de catapulte enregistrées")
    elif exp_type == "Mouvement plan général":
        st.markdown("### 📐 Données — Mouvement plan")

        t_list, x_list, y_list = [], [], []

        for i in range(n):
            c1, c2, c3 = st.columns(3)
            with c1:
                t = st.number_input(f"t[{i}] (s)", key=f"t_plan_{i}")
            with c2:
                x = st.number_input(f"x[{i}] (m)", key=f"x_plan_{i}")
            with c3:
                y = st.number_input(f"y[{i}] (m)", key=f"y_plan_{i}")

            t_list.append(t)
            x_list.append(x)
            y_list.append(y)

        if st.button("📤 Enregistrer l’expérience"):
            supabase.table("cinematique_2D").insert({
                "type": "plan",
                "results": {
                    "t": t_list,
                    "x": x_list,
                    "y": y_list
                }
            }).execute()

            st.success("✅ Données planaires enregistrées")
