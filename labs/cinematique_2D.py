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

    st.title("🧭 Laboratoire Cinématique 2D — Cégep Montmorency")
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

    theta = None
    if exp_type == "Projectile / Catapulte":
        angle_known = st.checkbox("Angle de lancement connu ?")
        if angle_known:
            theta = st.number_input("Angle θ (degrés)", value=45.0)

    if st.button("📤 Enregistrer l’expérience"):
        supabase.table("cinematique_2d").insert({
            "type": exp_type.lower().replace(" ", "_"),
            "results": {
                "t": t_list,
                "x": x_list,
                "y": y_list,
                "theta": theta
            }
        }).execute()
        st.success("✅ Données enregistrées sur Supabase")

    st.divider()

    # =======================
    # 3️⃣ Récupérer et analyser les données
    # =======================
    st.header("3️⃣ Analyse automatique")
    response = supabase.table("cinematique_2d").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    for sim in simulations:
        sim_id = sim["id"]
        st.markdown(f"## Simulation {sim_id} — {sim['created_at']}")
        df = pd.DataFrame(sim["results"])
        df = df.sort_values("t")
        t_vals = np.array(df["t"])
        x_vals = np.array(df["x"])
        y_vals = np.array(df["y"])

        # ---- Ajustement linéaire pour x ----
        ax, bx = np.polyfit(t_vals, x_vals, 1)
        x_fit = ax * t_vals + bx
        r2_x = 1 - np.sum((x_vals - x_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)

        # ---- Ajustement quadratique pour y ----
        ay, by, cy = np.polyfit(t_vals, y_vals, 2)
        y_fit = ay * t_vals**2 + by * t_vals + cy
        r2_y = 1 - np.sum((y_vals - y_fit)**2) / np.sum((y_vals - np.mean(y_vals))**2)

        # ---- Graphique trajectoire ----
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        x_smooth = ax * t_smooth + bx
        y_smooth = ay * t_smooth**2 + by * t_smooth + cy

        fig, ax_plot = plt.subplots(figsize=(6, 4))
        ax_plot.scatter(x_vals, y_vals, color="#1f2937", s=25, label="Données expérimentales")
        ax_plot.plot(x_smooth, y_smooth, color="crimson", linestyle="--", linewidth=2, label="Fit quadratique")
        ax_plot.set_xlabel("x (m)")
        ax_plot.set_ylabel("y (m)")
        ax_plot.set_title("Trajectoire du projectile")
        ax_plot.grid(True, linestyle="--", alpha=0.4)
        ax_plot.text(0.05, 0.05, f"R² x: {r2_x:.3f}\nR² y: {r2_y:.3f}",
                     transform=ax_plot.transAxes, fontsize=10,
                     bbox=dict(facecolor="white", alpha=0.5))
        ax_plot.legend(frameon=False)
        st.pyplot(fig)

        # ---- Extraction g expérimental ----
        g_exp = -2 * ay
        st.markdown("### ⚡ Extraction des paramètres physiques")
        st.write(f"v0x_exp = {ax:.3f} m/s, v0y_exp = {by:.3f} m/s, y0_exp = {cy:.3f} m, g_exp = {g_exp:.3f} m/s²")

        # ---- Calculs détaillés avec différentiation ----
        st.subheader("📐 Calculs différentielles détaillés")
        st.markdown("**Position :**")
        st.latex(r"\frac{dx}{dt} = v_x \implies x(t) = v_{0x} t + x_0")
        st.latex(r"\frac{dy}{dt} = v_y = v_{0y} - g t \implies y(t) = v_{0y} t - \frac{1}{2} g t^2 + y_0")

        st.markdown("**Fit expérimental :**")
        st.latex(rf"x(t) = {ax:.3f} t + {bx:.3f}")
        st.latex(rf"y(t) = {ay:.3f} t^2 + {by:.3f} t + {cy:.3f}")
        st.latex(rf"a_y = 2 * {ay:.3f} = {g_exp:.3f} m/s²")

        # ---- Substitution pour un temps spécifique
        st.subheader("⏱ Calculs pour un temps spécifique")
        t_input = st.number_input("Entrer un temps t (s)", value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")

        x_t = ax * t_input + bx
        y_t = ay * t_input**2 + by * t_input + cy
        vx_t = ax
        vy_t = 2 * ay * t_input + by
        ay_t = 2 * ay

        st.latex(rf"x({t_input}) = {ax:.3f} * {t_input} + {bx:.3f} = {x_t:.3f}")
        st.latex(rf"y({t_input}) = {ay:.3f} * {t_input}^2 + {by:.3f} * {t_input} + {cy:.3f} = {y_t:.3f}")
        st.latex(rf"v_x({t_input}) = {vx_t:.3f} m/s")
        st.latex(rf"v_y({t_input}) = 2 * {ay:.3f} * {t_input} + {by:.3f} = {vy_t:.3f} m/s")
        st.latex(rf"a_y({t_input}) = {ay_t:.3f} m/s²")
