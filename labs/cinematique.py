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

    n = st.number_input(
        "Nombre de mesures à entrer",
        min_value=2,
        max_value=50,
        value=5,
        step=1
    )

    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"x[{i}] (m)", value=0.0, key=f"x{i}")
        with col2:
            t_val = st.number_input(f"t[{i}] (s)", value=0.0, key=f"t{i}")
        x_list.append(x_val)
        t_list.append(t_val)

    if st.button("Envoyer les données brutes"):
        data_brute = {"x": x_list, "t": t_list}
        supabase.table("cinematique_brute").insert(
            {"results": data_brute}
        ).execute()
        st.success("Données brutes envoyées sur Supabase ✅")

    # =======================
    # Gestion des simulations
    # =======================
    st.subheader("Modifier / Supprimer des simulations existantes")

    response = supabase.table("cinematique_brute").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation trouvée.")
        return

    sim_options = [
        f"Simulation {sim['id']} — {sim['created_at']}"
        for sim in simulations
    ]

    selected_sim = st.selectbox(
        "Choisissez une simulation à modifier",
        sim_options
    )

    sim_index = sim_options.index(selected_sim)
    sim_data = simulations[sim_index]

    df = pd.DataFrame(sim_data["results"])
    edited_df = st.data_editor(df, num_rows="dynamic")

    if st.button("Sauvegarder les modifications"):
        updated_data = edited_df.to_dict(orient="list")
        supabase.table("cinematique_brute").update(
            {"results": updated_data}
        ).eq("id", sim_data["id"]).execute()
        st.success("Données mises à jour ✅")

    if st.button("Supprimer cette simulation"):
        supabase.table("cinematique_brute").delete().eq(
            "id", sim_data["id"]
        ).execute()
        st.success("Simulation supprimée ✅")
        st.experimental_rerun()

    # =======================
# Graphiques et approximations (GUI moderne)
# =======================
st.subheader("Graphiques et approximations (moindres carrés)")

for sim in simulations:
    st.markdown(f"### Simulation {sim['id']}")

    df_sim = pd.DataFrame(sim["results"]).sort_values("t")

    t = df_sim["t"].values
    x = df_sim["x"].values

    # Domaine continu pour courbes lisses
    t_smooth = np.linspace(t.min(), t.max(), 300)

    # Régression linéaire
    a1, b1 = np.polyfit(t, x, 1)
    x_lin_smooth = a1 * t_smooth + b1

    # Régression quadratique
    a2, b2, c2 = np.polyfit(t, x, 2)
    x_quad_smooth = a2 * t_smooth**2 + b2 * t_smooth + c2

    # --------- STYLE MODERNE ---------
    fig, ax = plt.subplots(figsize=(6, 4))  # graphique plus compact

    # Points expérimentaux (plus petits, sobres)
    ax.scatter(
        t, x,
        s=25,
        color="#1f2937",   # gris foncé moderne
        alpha=0.8,
        label="Données expérimentales",
        zorder=3
    )

    # Approximation linéaire (fine)
    ax.plot(
        t_smooth, x_lin_smooth,
        linewidth=2,
        color="#2563eb",
        label="Approximation linéaire"
    )

    # Approximation quadratique (smooth + élégante)
    ax.plot(
        t_smooth, x_quad_smooth,
        linewidth=2.5,
        color="#dc2626",
        linestyle="-",
        label="Approximation quadratique"
    )

    # Axes & style
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Cinématique 1D : x(t)", fontsize=12)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)

    # Boîte d'équations (clean)
    eq_text = (
        f"x(t) ≈ {a1:.3f} t + {b1:.3f}\n"
        f"x(t) ≈ {a2:.3f} t² + {b2:.3f} t + {c2:.3f}"
    )

    ax.text(
        0.02, 0.96,
        eq_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        )
    )

    st.pyplot(fig)

