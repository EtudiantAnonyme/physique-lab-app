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
    # 3️⃣ Analyse automatique et paramètres
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

        # Vérifier si un angle de lancement est fourni
        theta = results.get("theta")
        if theta is not None:
            theta_rad = np.deg2rad(theta)
            delta_x = x_vals[-1] - x_vals[0]
            delta_t = t_vals[-1] - t_vals[0]
            if delta_t == 0:
                st.warning(f"Simulation {sim_id} : delta_t = 0, impossible de calculer v0")
                continue
            v0 = delta_x / (delta_t * np.cos(theta_rad))
            v0x = v0 * np.cos(theta_rad)
            v0y = v0 * np.sin(theta_rad)
            g_exp = -2 * ((y_vals[-1] - y_vals[0] - v0y * delta_t) / (delta_t**2))
            cy = y_vals[0]
            ay = -g_exp / 2
            by = v0y
            ax = v0x
            bx = x_vals[0]
        else:
            # Ajustements classiques si pas d'angle
            ax, bx = np.polyfit(t_vals, x_vals, 1)   # Linéaire x(t)
            ay, by, cy = np.polyfit(t_vals, y_vals, 2)  # Quadratique y(t)
            g_exp = -2 * ay
            v0x = ax
            v0y = by

        # ---- Graphique trajectoire ----
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        if theta is not None:
            x_smooth = v0x * t_smooth + bx
            y_smooth = v0y * t_smooth - 0.5 * g_exp * t_smooth**2 + cy
        else:
            x_smooth = ax * t_smooth + bx
            y_smooth = ay * t_smooth**2 + by * t_smooth + cy

        r2_x = 1 - np.sum((x_vals - x_smooth[:len(t_vals)])**2) / np.sum((x_vals - np.mean(x_vals))**2)
        r2_y = 1 - np.sum((y_vals - y_smooth[:len(t_vals)])**2) / np.sum((y_vals - np.mean(y_vals))**2)

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

        # ---- Paramètres physiques expérimentaux ----
        st.markdown("### ⚡ Paramètres physiques expérimentaux (LaTeX)")
        st.latex(rf"v_{{0x}} = {v0x:.3f} \ \mathrm{{m/s}},\quad "
                 rf"v_{{0y}} = {v0y:.3f} \ \mathrm{{m/s}},\quad "
                 rf"y_0 = {cy:.3f} \ \mathrm{{m}},\quad "
                 rf"g = {g_exp:.3f} \ \mathrm{{m/s^2}}")

        # ---- Calcul différentiel détaillé ----
        st.subheader("📐 Calculs différentielles détaillés")
        st.markdown("**Position et vitesse théorique :**")
        st.latex(r"\frac{dx}{dt} = v_x \implies x(t) = v_{0x} t + x_0")
        st.latex(r"\frac{dy}{dt} = v_y = v_{0y} - g t \implies y(t) = v_{0y} t - \frac{1}{2} g t^2 + y_0")

        st.markdown("**Transformation des données expérimentales en courbe :**")
        st.markdown(r"""
        On dispose d'une série de points expérimentaux $(t_i, x_i, y_i)$.  
        Pour $x(t)$, un fit linéaire est utilisé : $x(t) \approx v_{0x} t + x_0$.  
        Pour $y(t)$, un fit quadratique est utilisé : $y(t) \approx a t^2 + b t + c$,  
        où $a = -\frac{g_\mathrm{exp}}{2}$, $b \approx v_{0y}$, $c = y_0$.  
        Les coefficients sont déterminés par régression polynomiale sur les mesures expérimentales.
        """)

        st.markdown("**Fit expérimental (résultats numériques) :**")
        st.latex(rf"x(t) = {ax:.3f} t + {bx:.3f}")
        st.latex(rf"y(t) = {ay:.3f} t^2 + {by:.3f} t + {cy:.3f}")
        st.latex(rf"a_y = 2 \cdot {ay:.3f} = {g_exp:.3f} \ \mathrm{{m/s^2}}")

        st.markdown("**💡 Interprétation :**")
        st.markdown(r"""
        - La pente de $x(t)$ donne $v_{0x}$.  
        - La pente initiale et la concavité de $y(t)$ donnent $v_{0y}$ et $g$.  
        - La transformation permet de visualiser une trajectoire continue à partir de points expérimentaux discrets.
        """)

        # ---- Substitution pour un temps spécifique
        st.subheader("⏱ Calcul pour un temps spécifique")
        t_input = st.number_input(f"Entrer un temps t (s) pour simulation {sim_id}",
                                  value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")

        if theta is not None:
            x_t = v0x * t_input + bx
            y_t = v0y * t_input - 0.5 * g_exp * t_input**2 + cy
            vx_t = v0x
            vy_t = v0y - g_exp * t_input
        else:
            x_t = ax * t_input + bx
            y_t = ay * t_input**2 + by * t_input + cy
            vx_t = ax
            vy_t = 2 * ay * t_input + by

        ay_t = 2 * ay

        st.latex(rf"x({t_input}) = {x_t:.3f} m")
        st.latex(rf"y({t_input}) = {y_t:.3f} m")
        st.latex(rf"v_x({t_input}) = {vx_t:.3f} m/s")
        st.latex(rf"v_y({t_input}) = {vy_t:.3f} m/s")
        st.latex(rf"a_y({t_input}) = {ay_t:.3f} m/s^2")

        # ---- Calculatrice interactive
        st.subheader("🧮 Calculatrice cinématique interactive")
        calc_option = st.selectbox(f"Choisir la conversion pour simulation {sim_id}", [
            "Temps → Vitesse", "Vitesse → Temps", "Temps → Position",
            "Position → Temps", "Vitesse → Position", "Position → Vitesse"
        ])
        input_val = st.number_input("Entrer la valeur connue", value=0.0, step=0.1, key=f"calc_{sim_id}")

        if calc_option == "Temps → Vitesse":
            st.write(f"v_x = {v0x:.3f} m/s, v_y = {v0y - g_exp*input_val:.3f} m/s")
        elif calc_option == "Vitesse → Temps":
            if g_exp != 0:
                t_sol = (v0y - input_val)/g_exp
                t_sol = t_sol if t_sol >= 0 else None
            else:
                t_sol = None
            st.write(f"Temps possible: {t_sol:.3f} s" if t_sol is not None else "Pas de solution physique")
        elif calc_option == "Temps → Position":
            x_pos = v0x * input_val
            y_pos = y0_exp + v0y * input_val - 0.5 * g_exp * input_val**2
            st.write(f"x = {x_pos:.3f} m, y = {y_pos:.3f} m")
        elif calc_option == "Position → Temps":
            coeffs = [-0.5*g_exp, v0y, cy - input_val]
            t_sols = np.roots(coeffs)
            t_sols = t_sols[np.isreal(t_sols)].real
            t_sols = t_sols[t_sols >= 0]
            st.write(f"Temps possible: {t_sols}")
        elif calc_option == "Vitesse → Position":
            t_val = (v0y - input_val)/g_exp if g_exp != 0 else 0
            x_pos = v0x * t_val
            y_pos = cy + v0y * t_val - 0.5 * g_exp * t_val**2
            st.write(f"x = {x_pos:.3f} m, y = {y_pos:.3f} m")
        elif calc_option == "Position → Vitesse":
            coeffs = [-0.5*g_exp, v0y, cy - input_val]
            t_sols = np.roots(coeffs)
            t_sols = t_sols[np.isreal(coeffs)].real
            vy_sols = 2*ay*t_sols + v0y
            st.write(f"v_y possible(s): {vy_sols}")
