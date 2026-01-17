import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase
import io
import json

def run_mouvement_circulaire_lab():
    st.set_page_config(
        page_title="Laboratoire Mouvement Circulaire",
        page_icon="🔄",
        layout="wide"
    )

    st.title("🔄 Laboratoire Mouvement Circulaire — Cégep Montmorency")
    st.markdown("""
    Cette section permet de :
    - Enregistrer et gérer des mesures expérimentales de mouvement circulaire
    - Calculer vitesse angulaire et linéaire, accélération centripète
    - Visualiser graphiquement la trajectoire circulaire
    - Tester différentes valeurs pour mieux comprendre les phénomènes
    """)
    st.divider()

    # =======================
    # 1️⃣ Type de mesure
    # =======================
    st.header("1️⃣ Paramètres du mouvement circulaire")
    radius = st.number_input("Rayon du cercle (m)", min_value=0.01, value=1.0, step=0.1)
    n_points = st.number_input("Nombre de mesures temporelles", min_value=2, max_value=100, value=10, step=1)

    t_list, theta_list = [], []
    st.header("2️⃣ Ajouter les données expérimentales")
    for i in range(n_points):
        col1, col2 = st.columns(2)
        with col1:
            t = st.number_input(f"t[{i}] (s)", key=f"t_{i}")
        with col2:
            theta = st.number_input(f"θ[{i}] (degrés)", key=f"theta_{i}")
        t_list.append(t)
        theta_list.append(theta)

    if st.button("📤 Enregistrer l’expérience"):
        supabase.table("circular_motion").insert({
            "results": {
                "t": t_list,
                "theta": theta_list,
                "radius": radius
            }
        }).execute()
        st.success("✅ Données enregistrées sur Supabase")
    st.divider()

    # =======================
    # 3️⃣ Analyse automatique
    # =======================
    st.header("3️⃣ Analyse automatique")
    response = supabase.table("circular_motion").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    for sim in simulations:
        sim_id = sim.get("id", "N/A")
        st.markdown(f"## Simulation {sim_id} — {sim.get('created_at', 'Inconnu')}")

        results = sim.get("results")
        if results is None or not isinstance(results, dict):
            st.warning(f"Simulation {sim_id} n'a pas de données valides.")
            continue

        t_vals = np.array(results.get("t", []))
        theta_vals = np.deg2rad(np.array(results.get("theta", [])))  # radians
        radius = results.get("radius", 1.0)

        if len(t_vals) < 2:
            st.warning(f"Simulation {sim_id} : pas assez de points pour analyse.")
            continue

        # Calcul de vitesse angulaire et linéaire
        omega_vals = np.gradient(theta_vals, t_vals)           # ω = dθ/dt
        v_vals = omega_vals * radius                            # v = ω * r
        a_c_vals = v_vals**2 / radius                           # a_c = v^2 / r

        # ---- Graphique
        fig, ax_plot = plt.subplots(figsize=(6,6))
        x_vals = radius * np.cos(theta_vals)
        y_vals = radius * np.sin(theta_vals)
        ax_plot.plot(x_vals, y_vals, 'o-', label="Position expérimentale")
        ax_plot.set_aspect('equal')
        ax_plot.set_xlabel("x (m)")
        ax_plot.set_ylabel("y (m)")
        ax_plot.set_title("Trajectoire circulaire")
        ax_plot.grid(True, linestyle="--", alpha=0.4)
        ax_plot.legend()
        st.pyplot(fig)

        # ---- Paramètres physiques
        st.markdown("### ⚡ Paramètres expérimentaux")
        st.latex(rf"R = {radius:.3f}\ \mathrm{{m}}")
        st.latex(r"\omega(t) = \frac{d\theta}{dt} \implies v(t) = \omega(t) R \implies a_c(t) = \frac{v^2}{R}")

        st.subheader("📈 Valeurs calculées")
        df = pd.DataFrame({
            "t (s)": t_vals,
            "θ (°)": np.rad2deg(theta_vals),
            "ω (rad/s)": omega_vals,
            "v (m/s)": v_vals,
            "a_c (m/s²)": a_c_vals
        })
        st.dataframe(df)

        # ---- Export JSON
        json_data = json.dumps({
            "t": t_vals.tolist(),
            "theta_deg": np.rad2deg(theta_vals).tolist(),
            "omega": omega_vals.tolist(),
            "v": v_vals.tolist(),
            "a_c": a_c_vals.tolist(),
            "radius": radius
        }, indent=4)
        st.download_button(
            label="Télécharger les données (JSON)",
            data=json_data,
            file_name=f"circular_motion_sim_{sim_id}.json",
            mime="application/json"
        )

        # ---- Calculatrice interactive
        st.subheader("🧮 Calculatrice circulaire")
        calc_option = st.selectbox("Choisir le calcul", [
            "Temps → Angle θ",
            "Angle θ → Temps",
            "Temps → Vitesse linéaire",
            "Temps → Accélération centripète",
            "Vitesse → Temps"
        ], key=f"calc_select_{sim_id}")
        input_val = st.number_input("Entrer la valeur", value=0.0, key=f"input_val_{sim_id}")

        if st.button(f"Calculer pour simulation {sim_id}", key=f"calc_button_{sim_id}"):
            if calc_option == "Temps → Angle θ":
                idx = (np.abs(t_vals - input_val)).argmin()
                st.write(f"θ ≈ {np.rad2deg(theta_vals[idx]):.3f}°")
            elif calc_option == "Angle θ → Temps":
                idx = (np.abs(theta_vals - np.deg2rad(input_val))).argmin()
                st.write(f"t ≈ {t_vals[idx]:.3f} s")
            elif calc_option == "Temps → Vitesse linéaire":
                idx = (np.abs(t_vals - input_val)).argmin()
                st.write(f"v ≈ {v_vals[idx]:.3f} m/s")
            elif calc_option == "Temps → Accélération centripète":
                idx = (np.abs(t_vals - input_val)).argmin()
                st.write(f"a_c ≈ {a_c_vals[idx]:.3f} m/s²")
            elif calc_option == "Vitesse → Temps":
                idx = (np.abs(v_vals - input_val)).argmin()
                st.write(f"t ≈ {t_vals[idx]:.3f} s, θ ≈ {np.rad2deg(theta_vals[idx]):.3f}°")
