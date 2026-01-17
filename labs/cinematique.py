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
        sim_id = sim["id"]
        st.markdown(f"## Simulation {sim_id} — {sim['created_at']}")

        df = pd.DataFrame(sim["results"])
        df = df.sort_values("t")

        t_vals = df["t"].values
        x_vals = df["x"].values

        # =======================
        # APPROXIMATIONS
        # =======================
        # linéaire
        a1, b1 = np.polyfit(t_vals, x_vals, 1)
        # quadratique
        a2, b2, c2 = np.polyfit(t_vals, x_vals, 2)

        # Générer des points smooth pour le graphique
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        x_lin_smooth = a1 * t_smooth + b1
        x_quad_smooth = a2 * t_smooth**2 + b2 * t_smooth + c2

        # =======================
        # GRAPHIQUE MODERNE
        # =======================
        fig, ax = plt.subplots(figsize=(6, 4))

        # points expérimentaux
        ax.scatter(t_vals, x_vals, s=25, color="#1f2937", label="Données expérimentales", zorder=3)
        # approx linéaire
        ax.plot(t_smooth, x_lin_smooth, linewidth=2, color="royalblue", label="Approximation linéaire")
        # approx quadratique
        ax.plot(t_smooth, x_quad_smooth, linewidth=2.5, color="crimson", linestyle="--", label="Approximation quadratique")

        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Position en fonction du temps")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)

        st.pyplot(fig)

        # =======================
        # AFFICHAGE DES FORMULES
        # =======================
        st.subheader("Modèle mathématique expérimental")
        st.latex(rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}")

        st.subheader("Calcul différentiel")
        st.markdown("**Vitesse (dérivée première)**")
        st.latex(r"v(t) = \frac{dx}{dt}")
        st.latex(rf"v(t) = \frac{{d}}{{dt}} ({a2:.3f} t^2 + {b2:.3f} t + {c2:.3f})")
        st.latex(rf"v(t) = {2*a2:.3f} t + {b2:.3f}")

        st.markdown("**Accélération (dérivée seconde)**")
        st.latex(r"a(t) = \frac{d^2x}{dt^2}")
        st.latex(rf"a(t) = \frac{{d}}{{dt}} ({2*a2:.3f} t + {b2:.3f})")
        st.latex(rf"a(t) = {2*a2:.3f}")

        # =======================
        # TABLEAU CINÉMATIQUE
        # =======================
        st.subheader("Tableau cinématique basé sur le modèle")
        v_model = 2 * a2 * t_vals + b2
        a_model = np.full_like(t_vals, 2 * a2)

        df_phys = pd.DataFrame({
            "t (s)": t_vals,
            "x(t) (m)": x_vals,
            "v(t) (m/s)": v_model,
            "a(t) (m/s²)": a_model
        })

        st.dataframe(df_phys, use_container_width=True)

        # =======================
        # CALCULATRICE CINÉMATIQUE
        # =======================
        st.subheader("Calculatrice : obtenir x, v, a pour un temps donné")

        t_input = st.number_input(
            "Entrer un temps t (s)",
            min_value=float(t_vals.min()),
            max_value=float(t_vals.max()),
            step=0.1,
            key=f"calc_{sim_id}"
        )

        x_calc = a2 * t_input**2 + b2 * t_input + c2
        v_calc = 2 * a2 * t_input + b2
        a_calc = 2 * a2

        col1, col2, col3 = st.columns(3)
        col1.metric("Position x(t)", f"{x_calc:.3f} m")
        col2.metric("Vitesse v(t)", f"{v_calc:.3f} m/s")
        col3.metric("Accélération a(t)", f"{a_calc:.3f} m/s²")

        st.divider()

        # =======================
        # MODIFIER / SUPPRIMER LES DONNÉES
        # =======================
        st.subheader("Modifier / Supprimer cette simulation")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            key=f"editor_{sim_id}"
        )

        # Sauvegarder
        if st.button("Sauvegarder les modifications", key=f"save_{sim_id}"):
            updated_data = edited_df.to_dict(orient="list")
            supabase.table("cinematique_brute").update({"results": updated_data}).eq("id", sim_id).execute()
            st.success("Simulation mise à jour ✅")
            st.experimental_rerun()

        # Supprimer
        if st.button("Supprimer cette simulation", key=f"delete_{sim_id}"):
            supabase.table("cinematique_brute").delete().eq("id", sim_id).execute()
            st.success("Simulation supprimée ✅")
            st.experimental_rerun()
