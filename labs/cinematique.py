import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase

def run_cinematique_lab():
    st.header("Laboratoire Cinématique 1D - Données brutes")

    # =======================
    # Ajouter de nouvelles mesures
    # =======================
    st.subheader("Ajouter de nouvelles mesures")
    st.write("Entrez vos mesures de position x (m) et temps t (s).")

    x_list = []
    t_list = []

    n = st.number_input("Nombre de mesures à entrer", min_value=2, max_value=50, value=5, step=1)

    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"x[{i}] (m)", value=0.0, key=f"x{i}")
        with col2:
            t_val = st.number_input(f"t[{i}] (s)", value=0.0, key=f"t{i}")
        x_list.append(x_val)
        t_list.append(t_val)

    if st.button("Envoyer les données brutes", key="send_new"):
        data_brute = {"x": x_list, "t": t_list}
        supabase.table("cinematique_brute").insert({"results": data_brute}).execute()
        st.success("Données brutes envoyées sur Supabase ! ✅")

    # =======================
    # Gérer les données existantes
    # =======================
    st.subheader("Modifier / Supprimer des simulations existantes")

    response = supabase.table("cinematique_brute").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation trouvée.")
        return

    # Sélectionner une simulation
    sim_options = [f"Simulation {sim['id']} - {sim['created_at']}" for sim in simulations]
    selected_sim = st.selectbox("Choisissez une simulation à modifier", sim_options)
    sim_index = sim_options.index(selected_sim)
    sim_data = simulations[sim_index]

    # Afficher les données dans un DataFrame éditable
    df = pd.DataFrame(sim_data["results"])
    edited_df = st.data_editor(df, num_rows="dynamic")  # permet modifier ou ajouter des lignes

    # Bouton pour sauvegarder les modifications
    if st.button("Sauvegarder les modifications", key="save_mod"):
        updated_data = edited_df.to_dict(orient="list")
        supabase.table("cinematique_brute").update({"results": updated_data}).eq("id", sim_data["id"]).execute()
        st.success("Données mises à jour avec succès ! ✅")

    # Bouton pour supprimer la simulation
    if st.button("Supprimer cette simulation", key="delete_sim"):
        supabase.table("cinematique_brute").delete().eq("id", sim_data["id"]).execute()
        st.success("Simulation supprimée ✅")
        st.session_state['rerun'] = True  # pour éviter l'erreur rerun

    # Rafraîchir la page après suppression
    if st.session_state.get('rerun', False):
        st.session_state['rerun'] = False
        st.experimental_rerun()

    # =======================
    # Graphiques et approximations
    # =======================
    st.subheader("Graphiques et approximations pour chaque simulation")

    for sim in simulations:
        st.markdown(f"### Simulation {sim['id']} - {sim['created_at']}")
        df_sim = pd.DataFrame(sim["results"])
        t_values = df_sim["t"].values
        x_values = df_sim["x"].values

        # Régression linéaire (degré 1)
        coeffs_lin = np.polyfit(t_values, x_values, 1)  # [a, b]
        x_lin = np.polyval(coeffs_lin, t_values)
        eq_lin = f"x(t) ≈ {coeffs_lin[0]:.3f}·t + {coeffs_lin[1]:.3f}"

        # Régression quadratique (degré 2)
        coeffs_quad = np.polyfit(t_values, x_values, 2)  # [a, b, c]
        x_quad = np.polyval(coeffs_quad, t_values)
        eq_quad = f"x(t) ≈ {coeffs_quad[0]:.3f}·t² + {coeffs_quad[1]:.3f}·t + {coeffs_quad[2]:.3f}"

        # Afficher les équations algébriques
        st.markdown(f"**Approximation linéaire :** {eq_lin}")
        st.markdown(f"**Approximation quadratique :** {eq_quad}")

        # Graphique matplotlib
        fig, ax = plt.subplots()
        ax.plot(t_values, x_values, "o", label="Données brutes")
        ax.plot(t_values, x_lin, "-", label="Approx. linéaire")
        ax.plot(t_values, x_quad, "--", label="Approx. quadratique")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Position en fonction du temps")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
