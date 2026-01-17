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
    - Ajuster des modèles linéaires et quadratiques
    - Visualiser graphiquement les trajectoires
    - Calculer vitesse et accélération
    - Tester différents temps ou valeurs pour comprendre les phénomènes cinématiques
    """)
    st.divider()

    # =======================
    # 1️⃣ Type d’expérience
    # =======================
    st.header("1️⃣ Type d’expérience")
    exp_type = st.selectbox(
        "Choisissez le type d’expérience",
        ["Projectile / Catapulte", "Mouvement plan général"]
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
        sim_id = sim.get("id", "N/A")
        sim_type = sim.get("type", "inconnu")
        st.markdown(f"## Simulation {sim_id} — {sim.get('created_at', 'Inconnu')} (Type: {sim_type})")

        results = sim.get("results")
        if results is None or not isinstance(results, dict):
            st.warning(f"Simulation {sim_id} n'a pas de données valides.")
            continue

        # DataFrame à partir du JSON
        df = pd.DataFrame({
            "t": results.get("t", []),
            "x": results.get("x", []),
            "y": results.get("y", [])
        })

        if df.empty:
            st.warning(f"Simulation {sim_id} n'a pas de mesures.")
            continue

        # Trier par temps
        df = df.sort_values("t")
        t_vals = np.array(df["t"])
        x_vals = np.array(df["x"])
        y_vals = np.array(df["y"])

        # ---- Ajustements ----
        ax, bx = np.polyfit(t_vals, x_vals, 1)   # Linéaire x(t)
        ay, by, cy = np.polyfit(t_vals, y_vals, 2)  # Quadratique y(t)

        x_fit = ax * t_vals + bx
        y_fit = ay * t_vals**2 + by * t_vals + cy

        r2_x = 1 - np.sum((x_vals - x_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)
        r2_y = 1 - np.sum((y_vals - y_fit)**2) / np.sum((y_vals - np.mean(y_vals))**2)

        # ---- Graphique ----
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        x_smooth = ax * t_smooth + bx
        y_smooth = ay * t_smooth**2 + by * t_smooth + cy

        fig, ax_plot = plt.subplots(figsize=(6, 4))
        ax_plot.scatter(x_vals, y_vals, color="#1f2937", s=25, label="Données expérimentales")
        ax_plot.plot(x_smooth, y_smooth, color="crimson", linestyle="--", linewidth=2, label="Fit quadratique")

        # ---- Analyse théorique si angle connu ----
        theta_deg = results.get("theta")
        if sim_type == "projectile/_catapulte" and theta_deg is not None:
            theta_rad = np.radians(theta_deg)
            v0x = ax
            v0y = v0x * np.tan(theta_rad)
            y0 = cy
            g_theo = -2 * ay

            x_theo = v0x * t_smooth
            y_theo = v0y * t_smooth - 0.5 * g_theo * t_smooth**2 + y0

            ax_plot.plot(x_theo, y_theo, color="green", linestyle="-.", linewidth=2, label="Théorie angle")
            st.markdown("### ⚡ Analyse théorique avec angle connu")
            st.write(f"Angle θ = {theta_deg}°")
            st.write(f"v0x théorique = {v0x:.3f} m/s, v0y théorique = {v0y:.3f} m/s, y0 = {y0:.3f} m, g = {g_theo:.3f} m/s²")

        ax_plot.set_xlabel("x (m)")
        ax_plot.set_ylabel("y (m)")
        ax_plot.set_title("Trajectoire du projectile")
        ax_plot.grid(True, linestyle="--", alpha=0.4)
        ax_plot.text(0.05, 0.05, f"R² x: {r2_x:.3f}\nR² y: {r2_y:.3f}",
                    transform=ax_plot.transAxes, fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.5))
        ax_plot.legend(frameon=False)
        st.pyplot(fig)

        # ---- Extraction des paramètres physiques ----
        g_exp = -2 * ay
        st.markdown("### ⚡ Paramètres physiques expérimentaux")
        st.write(f"v0x_exp = {ax:.3f} m/s, v0y_exp = {by:.3f} m/s, y0_exp = {cy:.3f} m, g_exp = {g_exp:.3f} m/s²")

        # ---- Calculs différentielles ----
        st.subheader("📐 Calculs différentielles détaillés")
        st.markdown("**Position et vitesse théorique :**")
        st.latex(r"\frac{dx}{dt} = v_x \implies x(t) = v_{0x} t + x_0")
        st.latex(r"\frac{dy}{dt} = v_y = v_{0y} - g t \implies y(t) = v_{0y} t - \frac{1}{2} g t^2 + y_0")

        st.markdown("**Fit expérimental :**")
        st.latex(rf"x(t) = {ax:.3f} t + {bx:.3f}")
        st.latex(rf"y(t) = {ay:.3f} t^2 + {by:.3f} t + {cy:.3f}")
        st.latex(rf"a_y = 2 * {ay:.3f} = {g_exp:.3f} m/s²")

        # ---- Calcul pour un temps spécifique
        st.subheader("⏱ Calcul pour un temps spécifique")
        t_input = st.number_input(f"Entrer un temps t (s) pour simulation {sim_id}", value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")

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
