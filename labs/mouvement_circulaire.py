import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    - Visualiser graphiquement la trajectoire circulaire (2D et 3D)
    - Tester différentes valeurs pour comprendre la cinématique circulaire
    - Bonus : animation de la particule en rotation
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
        # Calcul automatique
        t_arr = np.array(t_list)
        theta_rad = np.deg2rad(np.array(theta_list))
        dt = np.diff(t_arr)
        dtheta = np.diff(theta_rad)
        omega = np.zeros(len(t_arr))
        omega[1:] = dtheta / dt
        omega[0] = omega[1]  # initial
        v = omega * radius
        a_c = v**2 / radius
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

        # ---- Tableau
        st.subheader("📊 Tableau des mesures")
        st.dataframe(df)

        # ---- Graphique 2D
        fig2d, ax2d = plt.subplots(figsize=(6,6))
        ax2d.plot(df["x"], df["y"], marker='o', linestyle='-', color='crimson')
        ax2d.set_xlabel("x (m)")
        ax2d.set_ylabel("y (m)")
        ax2d.set_title("Trajectoire circulaire 2D")
        ax2d.set_aspect('equal', 'box')
        ax2d.grid(True)
        st.pyplot(fig2d)

        # ---- Graphique 3D interactif smooth
        st.subheader("🌐 Trajectoire 3D interactive (x, y, t)")

        # Interpolation pour une courbe lisse
        t_smooth = np.linspace(df["t"].min(), df["t"].max(), 300)
        theta_smooth = np.interp(t_smooth, df["t"], np.deg2rad(df["theta"]))
        r_smooth = results.get("radius", 1.0)
        x_smooth = r_smooth * np.cos(theta_smooth)
        y_smooth = r_smooth * np.sin(theta_smooth)
        z_smooth = t_smooth

        # Gradient de couleur selon le temps
        colors = z_smooth

        # Plotly 3D
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x_smooth,
            y=y_smooth,
            z=z_smooth,
            mode='lines+markers',
            marker=dict(size=4, color=colors, colorscale='Viridis', colorbar=dict(title='Temps (s)')),
            line=dict(color='blue', width=4)
        )])
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='x (m)',
                yaxis_title='y (m)',
                zaxis_title='t (s)',
                aspectmode='auto'
            ),
            title="Trajectoire circulaire 3D smooth",
            width=700,
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # ---- Téléchargement CSV / JSON / graphiques
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

        buf_img = io.BytesIO()
        fig2d.savefig(buf_img, format='png')
        buf_img.seek(0)
        st.download_button(
            "Télécharger graphique 2D",
            data=buf_img,
            file_name=f"simulation_{sim_id}_trajectory.png",
            mime="image/png"
        )

        # ---- Calculatrice cinématique avancée
        st.subheader("🧮 Calculatrice cinématique")
        calc_option = st.selectbox(f"Calcul pour simulation {sim_id}", [
            "Temps → Vitesse tangente",
            "Vitesse → Temps",
            "Temps → Position (x, y)",
            "Position → Temps",
            "Position → Vitesse tangente",
            "Angle → Vitesse"
        ], key=f"calc_{sim_id}")

        pos_axis = "x"
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
                
        # ---- Surprise 🎁 : Animation circulaire physiquement correcte
        st.subheader("🎬 Animation du mouvement circulaire (physiquement correcte)")

        from scipy.interpolate import CubicSpline
        import plotly.graph_objects as go

        # ======================
        # Données expérimentales
        # ======================
        t_arr = np.array(df["t"])
        theta_deg = np.array(df["theta"])
        r = results.get("radius", 1.0)

        # Sécurité : ordre temporel
        idx = np.argsort(t_arr)
        t_arr = t_arr[idx]
        theta_deg = theta_deg[idx]

        # Conversion radians
        theta_rad = np.deg2rad(theta_deg)

        # ======================
        # Interpolation angulaire
        # ======================
        t_smooth = np.linspace(t_arr.min(), t_arr.max(), 600)
        cs_theta = CubicSpline(t_arr, theta_rad)
        theta_smooth = cs_theta(t_smooth)

        # ======================
        # Reconstruction circulaire
        # ======================
        x_smooth = r * np.cos(theta_smooth)
        y_smooth = r * np.sin(theta_smooth)

        # ======================
        # Limites propres
        # ======================
        margin = 0.25 * r
        x_min, x_max = -r - margin, r + margin
        y_min, y_max = -r - margin, r + margin

        # ======================
        # Figure Plotly
        # ======================
        fig_anim = go.Figure()

        # Cercle complet (référence)
        theta_ref = np.linspace(0, 2*np.pi, 400)
        fig_anim.add_trace(go.Scatter(
            x=r*np.cos(theta_ref),
            y=r*np.sin(theta_ref),
            mode="lines",
            line=dict(color="lightgray", width=2, dash="dot"),
            name="Cercle théorique"
        ))

        # Trajectoire parcourue + particule
        fig_anim.add_trace(go.Scatter(
            x=[x_smooth[0]],
            y=[y_smooth[0]],
            mode="lines+markers",
            line=dict(color="crimson", width=3),
            marker=dict(size=10),
            name="Particule"
        ))

        # ======================
        # Frames
        # ======================
        frames = [
            go.Frame(
                data=[
                    fig_anim.data[0],
                    go.Scatter(
                        x=x_smooth[:k+1],
                        y=y_smooth[:k+1]
                    )
                ],
                name=str(k)
            )
            for k in range(len(t_smooth))
        ]

        fig_anim.frames = frames

        # ======================
        # Layout & animation
        # ======================
        fig_anim.update_layout(
            title="Animation du mouvement circulaire — rayon constant",
            xaxis=dict(range=[x_min, x_max], title="x (m)", scaleanchor="y"),
            yaxis=dict(range=[y_min, y_max], title="y (m)"),
            height=520,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 20, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate"
                        }]
                    )
                ]
            )]
        )

        st.plotly_chart(fig_anim, use_container_width=True)


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
