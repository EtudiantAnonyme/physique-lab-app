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
        page_icon="🌀",
        layout="wide"
    )

    st.title("🌀 Laboratoire Mouvement Circulaire — Cégep Montmorency")
    st.markdown("""
    Cette application permet de :
    - Enregistrer et gérer des mesures expérimentales en rotation
    - Calculer automatiquement ω, vitesse, accélération centripète
    - Visualiser graphiquement la trajectoire circulaire
    - Tester différentes valeurs pour comprendre la cinématique circulaire
    """)
    st.divider()

    # =======================
    # 1️⃣ Type d’expérience
    # =======================
    st.header("1️⃣ Type d’expérience")
    exp_type = st.selectbox(
        "Choisissez le type d’expérience",
        ["Rotation uniforme", "Rotation accélérée"]
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

    t_list, theta_list = [], []
    for i in range(n):
        c1, c2 = st.columns(2)
        with c1:
            t = st.number_input(f"t[{i}] (s)", key=f"t_{i}")
        with c2:
            theta = st.number_input(f"θ[{i}] (degrés)", key=f"theta_{i}")
        t_list.append(t)
        theta_list.append(theta)

    radius = st.number_input("Rayon de rotation r (m)", min_value=0.01, value=1.0, step=0.01)

    if st.button("📤 Enregistrer l’expérience"):
        # Calcul automatique des paramètres
        t_arr = np.array(t_list)
        theta_rad = np.deg2rad(np.array(theta_list))
        dt = np.diff(t_arr)
        dtheta = np.diff(theta_rad)

        # ω moyen (si dt non nul)
        omega = np.zeros(len(t_arr))
        omega[1:] = dtheta / dt
        omega[0] = omega[1]  # valeur initiale

        # Vitesse tangente
        v = omega * radius

        # Accélération centripète
        a_c = v**2 / radius

        # Coordonnées x, y
        x = radius * np.cos(theta_rad)
        y = radius * np.sin(theta_rad)

        results = {
            "t": t_list,
            "theta": theta_list,
            "radius": radius,
            "omega": omega.tolist(),
            "v": v.tolist(),
            "a_c": a_c.tolist(),
            "x": x.tolist(),
            "y": y.tolist()
        }

        supabase.table("mouvement_circulaire").insert({
            "type": exp_type.lower().replace(" ", "_"),
            "results": results
        }).execute()
        st.success("✅ Données enregistrées sur Supabase")

    st.divider()

    # =======================
    # 3️⃣ Analyse automatique
    # =======================
    st.header("3️⃣ Analyse automatique")
    response = supabase.table("mouvement_circulaire").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    for sim in simulations:
        sim_id = sim.get("id", "N/A")
        sim_type = sim.get("type", "inconnu")
        st.markdown(f"## Simulation {sim_id} — Type : {sim_type}")

        results = sim.get("results")
        if results is None or not isinstance(results, dict):
            st.warning(f"Simulation {sim_id} n'a pas de données valides.")
            continue

        # Création du DataFrame
        df = pd.DataFrame({
            "t": results.get("t", []),
            "theta": results.get("theta", []),
            "omega": results.get("omega", []),
            "v": results.get("v", []),
            "a_c": results.get("a_c", []),
            "x": results.get("x", []),
            "y": results.get("y", [])
        })

        if df.empty:
            st.warning(f"Simulation {sim_id} n'a pas de mesures.")
            continue

        st.subheader("📊 Tableau des mesures")
        st.dataframe(df)

        # ---- Graphique de la trajectoire
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(df["x"], df["y"], marker='o', linestyle='-', color='crimson')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Trajectoire circulaire")
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        st.pyplot(fig)

        # Téléchargement CSV/JSON
        buf_csv = io.StringIO()
        df.to_csv(buf_csv, index=False)
        st.download_button(
            "Télécharger données CSV",
            data=buf_csv.getvalue(),
            file_name=f"simulation_{sim_id}.csv",
            mime="text/csv"
        )

        json_data = json.dumps(results, indent=4)
        st.download_button(
            "Télécharger données JSON",
            data=json_data,
            file_name=f"simulation_{sim_id}.json",
            mime="application/json"
        )

        # Téléchargement graphique
        buf_img = io.BytesIO()
        fig.savefig(buf_img, format='png')
        buf_img.seek(0)
        st.download_button(
            "Télécharger graphique",
            data=buf_img,
            file_name=f"simulation_{sim_id}_trajectory.png",
            mime="image/png"
        )

        # ---- Calculatrice cinématique
        st.subheader("🧮 Calculatrice cinématique")
        calc_option = st.selectbox(f"Calcul pour simulation {sim_id}", [
            "Temps → Vitesse tangente",
            "Vitesse → Temps",
            "Temps → Position (x, y)",
            "Position → Temps",
            "Position → Vitesse tangente",
            "Angle → Vitesse"
        ], key=f"calc_{sim_id}")

        pos_axis = "x"  # par défaut
        if calc_option.startswith("Position →"):
            pos_axis = st.radio("Position par rapport à :", ["x", "y"], key=f"pos_axis_{sim_id}")

        input_val = st.number_input("Entrer la valeur", value=0.0, key=f"input_{sim_id}")

        if st.button(f"Calculer", key=f"calc_btn_{sim_id}"):
            r = results["radius"]
            omega_arr = np.array(results["omega"])
            v_arr = np.array(results["v"])
            t_arr = np.array(results["t"])
            x_arr = np.array(results["x"])
            y_arr = np.array(results["y"])
            theta_arr = np.deg2rad(np.array(results["theta"]))

            if calc_option == "Temps → Vitesse tangente":
                v_val = np.interp(input_val, t_arr, v_arr)
                st.write(f"v ≈ {v_val:.3f} m/s")

            elif calc_option == "Vitesse → Temps":
                idx = (np.abs(v_arr - input_val)).argmin()
                st.write(f"t ≈ {t_arr[idx]:.3f} s")

            elif calc_option == "Temps → Position (x, y)":
                x_val = np.interp(input_val, t_arr, x_arr)
                y_val = np.interp(input_val, t_arr, y_arr)
                st.write(f"x ≈ {x_val:.3f} m, y ≈ {y_val:.3f} m")

            elif calc_option == "Position → Temps":
                # Trouver le temps correspondant à la position choisie
                arr = x_arr if pos_axis == "x" else y_arr
                idx = (np.abs(arr - input_val)).argmin()
                st.write(f"t ≈ {t_arr[idx]:.3f} s")

            elif calc_option == "Position → Vitesse tangente":
                arr = x_arr if pos_axis == "x" else y_arr
                idx = (np.abs(arr - input_val)).argmin()
                st.write(f"v ≈ {v_arr[idx]:.3f} m/s")

            elif calc_option == "Angle → Vitesse":
                theta_deg = input_val
                idx = (np.abs(np.array(results["theta"]) - theta_deg)).argmin()
                st.write(f"v ≈ {v_arr[idx]:.3f} m/s, ω ≈ {omega_arr[idx]:.3f} rad/s")

    st.divider()

    # =======================
    # 4️⃣ Gestion des expériences enregistrées
    # =======================
    st.header("4️⃣ Gestion des expériences")
    for sim in simulations:
        sim_id = sim.get("id", "N/A")
        st.markdown(f"**Simulation {sim_id}**")

        if st.button(f"Supprimer simulation {sim_id}", key=f"del_{sim_id}"):
            supabase.table("mouvement_circulaire").delete().eq("id", sim_id).execute()
            st.success(f"✅ Simulation {sim_id} supprimée")
            st.experimental_rerun()
