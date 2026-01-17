import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase


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
    - Étudier les trajectoires de projectiles
    - Préparer les bases pour la dynamique 2D
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
            "Mouvement plan général"
        ]
    )

    st.divider()

    # =======================
    # 2️⃣ Données expérimentales
    # =======================
    st.header("2️⃣ Ajouter des données expérimentales")

    n = st.number_input(
        "Nombre de mesures",
        min_value=2,
        max_value=50,
        value=10,
        step=1
    )

    t_list, x_list, y_list = [], [], []

    for i in range(n):
        c1, c2, c3 = st.columns(3)
        with c1:
            t = st.number_input(f"t[{i}] (s)", key=f"t_{i}")
        with c2:
            x = st.number_input(f"x[{i}] (m)", key=f"x_{i}")
        with c3:
            y = st.number_input(f"y[{i}] (m)", key=f"y_{i}")

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)

    angle = None
    if exp_type == "Projectile / Catapulte":
        angle = st.number_input("Angle de lancement θ (degrés)", value=45.0)

    # =======================
    # 3️⃣ Enregistrement Supabase
    # =======================
    if st.button("📤 Enregistrer l’expérience"):
        try:
            rows = []
            for t, x, y in zip(t_list, x_list, y_list):
                rows.append({
                    "experience_type": exp_type,
                    "angle": angle,
                    "temps": t,
                    "distance_x": x,
                    "distance_y": y
                })

            supabase.table("cinematique_2d").insert(rows).execute()

            st.success("✅ Données enregistrées dans Supabase")

        except Exception as e:
            st.error("❌ Erreur lors de l'enregistrement")
            st.exception(e)
