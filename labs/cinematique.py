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
# Graphiques et approximations (version améliorée)
# =======================
st.subheader("Graphiques et approximations (moindres carrés)")

for sim in simulations:
    st.markdown(f"### Simulation {sim['id']}")

    df_sim = pd.DataFrame(sim["results"])
    df_sim = df_sim.sort_values("t")  # IMPORTANT en physique

    t = df_sim["t"].values
    x = df_sim["x"].values

    # Régression linéaire
    a1, b1 = np.polyfit(t, x, 1)
    x_lin = a1 * t + b1

    # Régression quadratique
    a2, b2, c2 = np.polyfit(t, x, 2)
    x_quad = a2 * t**2 + b2 * t + c2

    # Création du graphique
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Données expérimentales
    ax.scatter(t, x, color="black", s=60, label="Données expérimentales", zorder=3)

    # Approximations
    ax.plot(t, x_lin, color="royalblue", linewidth=2,
            label="Approximation linéaire")
    ax.plot(t, x_quad, color="crimson", linestyle="--", linewidth=2,
            label="Approximation quadratique")

    # Axes et style
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Cinématique 1D : position en fonction du temps")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Équations algébriques (dans une boîte)
    eq_text = (
        f"Linéaire : x(t) ≈ {a1:.3f} t + {b1:.3f}\n"
        f"Quadratique : x(t) ≈ {a2:.3f} t² + {b2:.3f} t + {c2:.3f}"
    )

    ax.text(
        0.02, 0.95, eq_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    st.pyplot(fig)
