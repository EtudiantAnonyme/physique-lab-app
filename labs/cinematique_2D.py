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
    n = st.number_input("Nombre de mesures", min_value=2, max_value=100, value=10, step=1)

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
        sim_id = sim.get("id", "N/A")
        sim_type = sim.get("type", "inconnu")
        st.markdown(f"## Simulation {sim_id} — {sim.get('created_at', 'Inconnu')} (Type: {sim_type})")

        results = sim.get("results")
        if results is None or not isinstance(results, dict):
            st.warning(f"Simulation {sim_id} n'a pas de données valides.")
            continue

        try:
            df = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Impossible de créer le tableau pour la simulation {sim_id}: {e}")
            continue

        if "t" in df.columns:
            df = df.sort_values("t")

        t_vals = np.array(df["t"])
        x_vals = np.array(df["x"])
        y_vals = np.array(df["y"])

        # =======================
        # 4️⃣ Ajustements selon le type d'expérience
        # =======================
        if sim_type == "projectile/_catapulte":
            # projectile : x linéaire, y quadratique
            ax, bx = np.polyfit(t_vals, x_vals, 1)
            ay, by, cy = np.polyfit(t_vals, y_vals, 2)
            x_fit = ax * t_vals + bx
            y_fit = ay * t_vals**2 + by * t_vals + cy
            r2_x = 1 - np.sum((x_vals - x_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)
            r2_y = 1 - np.sum((y_vals - y_fit)**2) / np.sum((y_vals - np.mean(y_vals))**2)

            g_exp = -2 * ay  # accélération gravitationnelle expérimentale

        elif sim_type == "mouvement_plan_général":
            # mouvement plan : x et y linéaires
            ax, bx = np.polyfit(t_vals, x_vals, 1)
            ay, by = np.polyfit(t_vals, y_vals, 1)
            x_fit = ax * t_vals + bx
            y_fit = ay * t_vals + by
            r2_x = 1 - np.sum((x_vals - x_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)
            r2_y = 1 - np.sum((y_vals - y_fit)**2) / np.sum((y_vals - np.mean(y_vals))**2)
            g_exp = 0

        else:
            st.info("Type d'expérience non supporté pour l'instant.")
            continue

        # =======================
        # 5️⃣ Graphique trajectoire
        # =======================
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        if sim_type == "projectile/_catapulte":
            x_smooth = ax * t_smooth + bx
            y_smooth = ay * t_smooth**2 + by * t_smooth + cy
        else:
            x_smooth = ax * t_smooth + bx
            y_smooth = ay * t_smooth + by

        fig, ax_plot = plt.subplots(figsize=(6, 4))
        ax_plot.scatter(x_vals, y_vals, color="#1f2937", s=25, label="Données expérimentales")
        ax_plot.plot(x_smooth, y_smooth, color="crimson", linestyle="--", linewidth=2, label="Fit")
        ax_plot.set_xlabel("x (m)")
        ax_plot.set_ylabel("y (m)")
        ax_plot.set_title("Trajectoire")
        ax_plot.grid(True, linestyle="--", alpha=0.4)
        ax_plot.text(0.05, 0.05, f"R² x: {r2_x:.3f}\nR² y: {r2_y:.3f}",
                     transform=ax_plot.transAxes, fontsize=10,
                     bbox=dict(facecolor="white", alpha=0.5))
        ax_plot.legend(frameon=False)
        st.pyplot(fig)

        # =======================
        # 6️⃣ Calculs détaillés
        # =======================
        st.subheader("📐 Calculs différentielles détaillés")
        if sim_type == "projectile/_catapulte":
            st.markdown("**Position et vitesse :**")
            st.latex(r"\frac{dx}{dt} = v_x \implies x(t) = v_{0x} t + x_0")
            st.latex(r"\frac{dy}{dt} = v_y = v_{0y} - g t \implies y(t) = v_{0y} t - \frac{1}{2} g t^2 + y_0")
            st.latex(rf"x(t) = {ax:.3f} t + {bx:.3f}")
            st.latex(rf"y(t) = {ay:.3f} t^2 + {by:.3f} t + {cy:.3f}")
            st.latex(rf"a_y = 2 * {ay:.3f} = {g_exp:.3f} m/s²")
        else:
            st.markdown("**Position et vitesse (linéaire) :**")
            st.latex(r"x(t) = v_x t + x_0")
            st.latex(r"y(t) = v_y t + y_0")
            st.latex(rf"x(t) = {ax:.3f} t + {bx:.3f}")
            st.latex(rf"y(t) = {ay:.3f} t + {by:.3f}")

        # =======================
        # 7️⃣ Calculs pour un temps spécifique
        # =======================
        st.subheader("⏱ Calculs pour un temps spécifique")
        t_input = st.number_input("Entrer un temps t (s)", value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")

        x_t = ax * t_input + bx
        y_t = ay * t_input**2 + by * t_input + cy if sim_type == "projectile/_catapulte" else ay * t_input + by
        vx_t = ax
        vy_t = 2 * ay * t_input + by if sim_type == "projectile/_catapulte" else ay
        ay_t = 2 * ay if sim_type == "projectile/_catapulte" else 0

        st.latex(rf"x({t_input}) = {x_t:.3f}")
        st.latex(rf"y({t_input}) = {y_t:.3f}")
        st.latex(rf"v_x({t_input}) = {vx_t:.3f} m/s")
        st.latex(rf"v_y({t_input}) = {vy_t:.3f} m/s")
        st.latex(rf"a_y({t_input}) = {ay_t:.3f} m/s²")
