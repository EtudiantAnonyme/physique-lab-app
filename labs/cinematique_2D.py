import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase
import io
import json

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
        st.experimental_rerun()

    st.divider()

    # =======================
    # 3️⃣ Analyse automatique
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

        # Créer DataFrame
        df = pd.DataFrame({
            "t": results.get("t", []),
            "x": results.get("x", []),
            "y": results.get("y", [])
        })

        if df.empty:
            st.warning(f"Simulation {sim_id} n'a pas de mesures.")
            continue

        df = df.sort_values("t")
        t_vals = np.array(df["t"])
        x_vals = np.array(df["x"])
        y_vals = np.array(df["y"])

        # Déterminer paramètres physiques
        theta_val = results.get("theta")
        if theta_val is not None:
            theta_rad = np.deg2rad(theta_val)
            delta_x = x_vals[-1] - x_vals[0]
            delta_t = t_vals[-1] - t_vals[0] if t_vals[-1] != t_vals[0] else 1e-6
            v0 = delta_x / (delta_t * np.cos(theta_rad))
            v0x = v0 * np.cos(theta_rad)
            v0y = v0 * np.sin(theta_rad)
            g_exp = -2 * ((y_vals[-1] - y_vals[0] - v0y*delta_t)/(delta_t**2))
            ax, bx = v0x, x_vals[0]
            ay, by, cy = -g_exp/2, v0y, y_vals[0]
        else:
            ax, bx = np.polyfit(t_vals, x_vals, 1)
            ay, by, cy = np.polyfit(t_vals, y_vals, 2)
            g_exp = -2*ay
            v0x, v0y = ax, by

        # ---- Trajectoire lisse pour le graphique
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        x_smooth = v0x * t_smooth + bx
        y_smooth = v0y * t_smooth + (-g_exp/2)*t_smooth**2 + cy

        # ---- Graphique
        fig, ax_plot = plt.subplots(figsize=(6,4))
        ax_plot.scatter(x_vals, y_vals, color="#1f2937", s=25, label="Données expérimentales")
        ax_plot.plot(x_smooth, y_smooth, color="crimson", linestyle="--", linewidth=2, label="Fit balistique")
        ax_plot.set_xlabel("x (m)")
        ax_plot.set_ylabel("y (m)")
        ax_plot.set_title("Trajectoire du projectile")
        ax_plot.grid(True, linestyle="--", alpha=0.4)
        r2_x = 1 - np.sum((x_vals - x_smooth[:len(x_vals)])**2)/np.sum((x_vals - np.mean(x_vals))**2)
        r2_y = 1 - np.sum((y_vals - y_smooth[:len(y_vals)])**2)/np.sum((y_vals - np.mean(y_vals))**2)
        ax_plot.text(0.05, 0.05, f"R² x: {r2_x:.3f}\nR² y: {r2_y:.3f}",
                     transform=ax_plot.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
        ax_plot.legend(frameon=False)
        st.pyplot(fig)

        # ---- Export CSV/JSON et téléchargement graphique
        st.subheader("💾 Exporter les données et graphique")
        df_export = pd.DataFrame({
            "t (s)": t_vals,
            "x (m)": x_vals,
            "y (m)": y_vals
        })
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name=f"simulation_{sim_id}_data.csv",
            mime="text/csv"
        )

        json_data = json.dumps(results, indent=4)
        st.download_button(
            label="Télécharger JSON",
            data=json_data,
            file_name=f"simulation_{sim_id}_data.json",
            mime="application/json"
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Télécharger graphique",
            data=buf,
            file_name=f"simulation_{sim_id}_trajectory.png",
            mime="image/png"
        )

        # ---- Paramètres physiques en LaTeX
        st.markdown("### ⚡ Paramètres physiques expérimentaux")
        st.latex(rf"v_{{0x}} = {v0x:.3f}\ \mathrm{{m/s}}, "
                 rf"v_{{0y}} = {v0y:.3f}\ \mathrm{{m/s}}, "
                 rf"y_0 = {cy:.3f}\ \mathrm{{m}}, "
                 rf"g = {g_exp:.3f}\ \mathrm{{m/s^2}}")

        # ---- Calcul différentiel
        st.subheader("📐 Calculs différentielles détaillés")
        st.markdown("**Position et vitesse théorique :**")
        st.latex(r"\frac{dx}{dt} = v_x \implies x(t) = v_{0x} t + x_0")
        st.latex(r"\frac{dy}{dt} = v_y = v_{0y} - g t \implies y(t) = v_{0y} t - \frac{1}{2} g t^2 + y_0")

        # ---- Calculatrice interactive
        st.subheader("🧮 Calculatrice cinématique")
        calc_option = st.selectbox("Choisir le calcul", [
            "Temps → Vitesse",
            "Vitesse → Temps",
            "Temps → Position",
            "Position → Temps",
            "Position → Vitesse",
            "Vitesse → Position"
        ])
        input_val = st.number_input("Entrer la valeur", value=0.0)

        if st.button(f"Calculer pour simulation {sim_id}", key=f"calc_{sim_id}"):

            if calc_option == "Temps → Vitesse":
                vx_calc = v0x
                vy_calc = v0y - g_exp*input_val
                st.write(f"v_x = {vx_calc:.3f} m/s, v_y = {vy_calc:.3f} m/s")

            elif calc_option == "Vitesse → Temps":
                if g_exp != 0:
                    t_calc = (v0y - input_val)/g_exp
                    st.write(f"t = {t_calc:.3f} s")
                else:
                    st.warning("g = 0, impossible de calculer le temps")

            elif calc_option == "Temps → Position":
                x_calc = v0x*input_val + bx
                y_calc = -0.5*g_exp*input_val**2 + v0y*input_val + cy
                st.write(f"x = {x_calc:.3f} m, y = {y_calc:.3f} m")

            elif calc_option == "Position → Temps":
                coeffs = [-0.5*g_exp, v0y, cy - input_val]
                t_sols = np.roots(coeffs)
                t_sols = t_sols[np.isreal(t_sols)].real
                st.write(f"Temps possibles: {t_sols}")

            elif calc_option == "Position → Vitesse":
                coeffs = [-0.5*g_exp, v0y, cy - input_val]
                t_sols = np.roots(coeffs)
                t_sols = t_sols[np.isreal(t_sols)].real
                vy_sols = v0y - g_exp*t_sols
                vx_calc = v0x
                st.write(f"Temps possibles: {t_sols}")
                st.write(f"v_x = {vx_calc:.3f} m/s, v_y correspondantes = {vy_sols}")

            elif calc_option == "Vitesse → Position":
                t_calc = (v0y - input_val)/g_exp if g_exp != 0 else 0
                y_calc = -0.5*g_exp*t_calc**2 + v0y*t_calc + cy
                x_calc = v0x * t_calc + bx
                st.write(f"x = {x_calc:.3f} m, y = {y_calc:.3f} m, t = {t_calc:.3f} s")

    st.divider()

    # =======================
    # 4️⃣ Gestion des données expérimentales
    # =======================
    st.header("4️⃣ Gestion des expériences enregistrées")

    for sim in simulations:
        sim_id = sim.get("id", "N/A")
        sim_type = sim.get("type", "inconnu")
        st.markdown(f"**Simulation {sim_id} — Type : {sim_type}**")

        # Supprimer
        if st.button(f"Supprimer simulation {sim_id}", key=f"del_{sim_id}"):
            supabase.table("cinematique_2d").delete().eq("id", sim_id).execute()
            st.success(f"✅ Simulation {sim_id} supprimée.")
            st.experimental_rerun()

        # Modifier
        if st.button(f"Modifier simulation {sim_id}", key=f"edit_{sim_id}"):
            results = sim.get("results", {})
            t_vals = results.get("t", [])
            x_vals = results.get("x", [])
            y_vals = results.get("y", [])
            theta = results.get("theta", None)

            st.markdown(f"### Modifier simulation {sim_id}")
            n_modify = st.number_input("Nombre de mesures", min_value=1, max_value=100, value=len(t_vals), key=f"n_mod_{sim_id}")

            t_new, x_new, y_new = [], [], []
            for i in range(n_modify):
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_i = st.number_input(f"t[{i}]", value=t_vals[i] if i < len(t_vals) else 0.0, key=f"t_mod_{sim_id}_{i}")
                with c2:
                    x_i = st.number_input(f"x[{i}]", value=x_vals[i] if i < len(x_vals) else 0.0, key=f"x_mod_{sim_id}_{i}")
                with c3:
                    y_i = st.number_input(f"y[{i}]", value=y_vals[i] if i < len(y_vals) else 0.0, key=f"y_mod_{sim_id}_{i}")
                t_new.append(t_i)
                x_new.append(x_i)
                y_new.append(y_i)

            theta_new = theta
            if "projectile" in sim_type:
                angle_known = st.checkbox("Angle de lancement connu ?", value=(theta is not None), key=f"theta_mod_{sim_id}")
                if angle_known:
                    theta_new = st.number_input("Angle θ (degrés)", value=theta if theta else 45.0, key=f"theta_val_{sim_id}")

            if st.button(f"📤 Enregistrer modifications simulation {sim_id}", key=f"save_mod_{sim_id}"):
                supabase.table("cinematique_2d").update({
                    "results": {"t": t_new, "x": x_new, "y": y_new, "theta": theta_new}
                }).eq("id", sim_id).execute()
                st.success(f"✅ Simulation {sim_id} mise à jour.")
                st.experimental_rerun()
