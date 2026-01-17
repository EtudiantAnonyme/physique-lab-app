import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase


def run_cinematique_lab():
    st.header("Laboratoire de cinématique 1D — Données brutes et analyse")

    # =======================
    # AJOUT DE DONNÉES BRUTES
    # =======================
    st.subheader("Ajouter des mesures expérimentales")

    n = st.number_input(
        "Nombre de mesures",
        min_value=2,
        max_value=50,
        value=5,
        step=1
    )

    x_list, t_list = [], []

    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input(f"x[{i}] (m)", key=f"x{i}")
        with col2:
            t = st.number_input(f"t[{i}] (s)", key=f"t{i}")
        x_list.append(x)
        t_list.append(t)

    if st.button("Enregistrer les données brutes"):
        supabase.table("cinematique_brute").insert({
            "results": {"t": t_list, "x": x_list}
        }).execute()
        st.success("Données enregistrées avec succès ✅")

    st.divider()

    # =======================
    # RÉCUPÉRATION DES DONNÉES
    # =======================
    response = supabase.table("cinematique_brute").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    # =======================
    # AFFICHAGE PAR SIMULATION
    # =======================
    for sim in simulations:
        st.markdown(f"## Simulation {sim['id']}")

        df = pd.DataFrame(sim["results"]).sort_values("t")

        t = df["t"].values
        x = df["x"].values

        # =======================
        # APPROXIMATIONS
        # =======================
        a1, b1 = np.polyfit(t, x, 1)
        a2, b2, c2 = np.polyfit(t, x, 2)

        t_smooth = np.linspace(t.min(), t.max(), 300)
        x_lin = a1 * t_smooth + b1
        x_quad = a2 * t_smooth**2 + b2 * t_smooth + c2

        # =======================
        # GRAPHIQUE MODERNE
        # =======================
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(t, x, s=25, color="#1f2937", label="Données expérimentales")
        ax.plot(t_smooth, x_lin, linewidth=2, label="Approximation linéaire")
        ax.plot(t_smooth, x_quad, linewidth=2.5, label="Approximation quadratique")

        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Position en fonction du temps")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)

        st.pyplot(fig)

        # =======================
        # MODÈLE MATHÉMATIQUE
        # =======================
        st.subheader("Modèle mathématique expérimental")

        st.latex(
            rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}"
        )

        # =======================
        # CALCUL DIFFÉRENTIEL
        # =======================
        st.subheader("Calcul différentiel")

        st.markdown("**Vitesse (dérivée première)**")
        st.latex(r"v(t) = \frac{dx}{dt}")
        st.latex(
            rf"v(t) = {2*a2:.3f} t + {b2:.3f}"
        )

        st.markdown("**Accélération (dérivée seconde)**")
        st.latex(r"a(t) = \frac{d^2x}{dt^2}")
        st.latex(
            rf"a(t) = {2*a2:.3f}"
        )

        # =======================
        # TABLEAU CINÉMATIQUE
        # =======================
        st.subheader("Tableau cinématique")

        v_vals = 2 * a2 * t + b2
        a_vals = np.full_like(t, 2 * a2)

        df_phys = pd.DataFrame({
            "Temps t (s)": t,
            "Position x(t) (m)": x,
            "Vitesse v(t) (m/s)": v_vals,
            "Accélération a(t) (m/s²)": a_vals
        })

        st.dataframe(df_phys, use_container_width=True)

        # =======================
        # CALCULATRICE PHYSIQUE
        # =======================
        st.subheader("Calculatrice cinématique")

        t_input = st.number_input(
            "Choisir un temps t (s)",
            min_value=float(t.min()),
            max_value=float(t.max()),
            step=0.1,
            key=f"calc_{sim['id']}"
        )

        x_calc = a2 * t_input**2 + b2 * t_input + c2
        v_calc = 2 * a2 * t_input + b2
        a_calc = 2 * a2

        col1, col2, col3 = st.columns(3)
        col1.metric("x(t)", f"{x_calc:.3f} m")
        col2.metric("v(t)", f"{v_calc:.3f} m/s")
        col3.metric("a(t)", f"{a_calc:.3f} m/s²")

        st.divider()
