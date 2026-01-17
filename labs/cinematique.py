import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase
import io

def run_cinematique_lab():
    st.set_page_config(
        page_title="Laboratoire Cinématique 1D",
        page_icon="🏃",
        layout="wide"
    )

    st.title("🧪 Laboratoire Cinématique 1D — Cégep Montmorency")
    st.markdown("""
    Cette application permet de :
    - Enregistrer et gérer des mesures expérimentales
    - Ajuster des modèles linéaires et quadratiques
    - Visualiser graphiquement les résultats
    - Calculer vitesse et accélération
    - Tester différents temps ou valeurs pour comprendre les phénomènes cinématiques
    """)

    st.divider()

    # =======================
    # 1️⃣ Ajouter des mesures expérimentales
    # =======================
    st.header("1️⃣ Ajouter des mesures expérimentales")
    n = st.number_input("Nombre de mesures", min_value=2, max_value=50, value=5, step=1)
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
        supabase.table("cinematique_brute").insert({"results": {"t": t_list, "x": x_list}}).execute()
        st.success("✅ Données enregistrées sur Supabase")

    st.divider()

    # =======================
    # 2️⃣ Gestion des simulations
    # =======================
    st.header("2️⃣ Gestion des simulations")
    response = supabase.table("cinematique_brute").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    # Sélectionner une simulation
    sim_options = [f"Simulation {sim['id']} — {sim['created_at']}" for sim in simulations]
    selected_sim = st.selectbox("Choisissez une simulation à analyser", sim_options)
    sim_index = sim_options.index(selected_sim)
    sim = simulations[sim_index]
    sim_id = sim["id"]

    df = pd.DataFrame(sim["results"])
    df = df.sort_values("t")
    t_vals = df["t"].values
    x_vals = df["x"].values

    # =======================
    # 3️⃣ Ajustements et graphiques
    # =======================
    st.header("3️⃣ Graphiques et ajustements")

    # Linéaire
    a1, b1 = np.polyfit(t_vals, x_vals, 1)
    x_lin_fit = a1 * t_vals + b1
    r2_lin = 1 - np.sum((x_vals - x_lin_fit)**2)/np.sum((x_vals - np.mean(x_vals))**2)

    # Quadratique
    a2, b2, c2 = np.polyfit(t_vals, x_vals, 2)
    x_quad_fit = a2*t_vals**2 + b2*t_vals + c2
    r2_quad = 1 - np.sum((x_vals - x_quad_fit)**2)/np.sum((x_vals - np.mean(x_vals))**2)

    # Lignes lisses
    t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
    x_lin_smooth = a1*t_smooth + b1
    x_quad_smooth = a2*t_smooth**2 + b2*t_smooth + c2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_vals, x_vals, s=20, color="#1f2937")
    ax.plot(t_smooth, x_lin_smooth, linewidth=1.5, color="royalblue", alpha=0.7, label="Approximation linéaire")
    ax.plot(t_smooth, x_quad_smooth, linewidth=1.5, color="red", alpha=0.9, label="Approximation quadratique")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position en fonction du temps")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    ax.text(
        0.95, 0.05,  # position (x=95% largeur, y=5% hauteur)
        f"R² linéaire = {r2_lin:.4f}\nR² quadratique = {r2_quad:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
    )

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # Bouton de téléchargement
    st.download_button(
        label="📥 Télécharger le graphique",
        data=buf,
        file_name=f"graphique_simulation_{sim_id}.png",
        mime="image/png"
    )
    st.markdown("**Qualité de l’ajustement (R²) :**")
    st.write(f"Linéaire : R² = {r2_lin:.4f}")
    st.write(f"Quadratique : R² = {r2_quad:.4f}")

    # =======================
    # 4️⃣ Modèle mathématique et dérivées
    # =======================
    st.header("4️⃣ Modèle mathématique et dérivées")
    st.markdown("Fonction quadratique ajustée :")
    st.latex(rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}")
    st.markdown("Vitesse et accélération à partir du modèle :")
    st.latex(r"v(t) = dx/dt = 2 a t + b")
    st.latex(rf"v(t) = {2*a2:.3f} t + {b2:.3f}")
    st.latex(r"a(t) = d^2x/dt^2 = 2 a")
    st.latex(rf"a(t) = {2*a2:.3f}")

    # =======================
    # 5️⃣ Tableau cinématique
    # =======================
    st.header("5️⃣ Tableau cinématique")
    v_model = 2 * a2 * t_vals + b2
    a_model = np.full_like(t_vals, 2*a2)
    df_phys = pd.DataFrame({
        "t (s)": t_vals,
        "x(t) (m)": x_vals,
        "v(t) (m/s)": v_model,
        "a(t) (m/s²)": a_model
    })
    st.dataframe(df_phys, use_container_width=True)

    # Copier les colonnes
    st.markdown("**📋 Copier les données d'une colonne :**")
    for col in df_phys.columns:
        col_data = ", ".join(df_phys[col].astype(str))
        st.text_area(f"Copier '{col}'", value=col_data, height=80)

    # =======================
    # 6️⃣ Calculatrice interactive
    # =======================
    st.header("6️⃣ Calculatrice interactive")
    t_input = st.number_input("Entrer un temps t (s)", value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")
    x_calc = a2*t_input**2 + b2*t_input + c2
    v_calc = 2*a2*t_input + b2
    a_calc = 2*a2
    col1, col2, col3 = st.columns(3)
    col1.metric("Position x(t)", f"{x_calc:.3f} m")
    col2.metric("Vitesse v(t)", f"{v_calc:.3f} m/s")
    col3.metric("Accélération a(t)", f"{a_calc:.3f} m/s²")

    st.markdown("**Explication des calculs :**")
    st.latex(rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}")
    st.latex(rf"v(t) = 2 \cdot {a2:.3f} t + {b2:.3f}")
    st.latex(rf"a(t) = 2 \cdot {a2:.3f}")

    # =======================
    # 7️⃣ Calculer t à partir de x ou v
    # =======================
    st.header("7️⃣ Calculer le temps à partir de x ou v")
    option = st.selectbox("Variable connue", ["Position x(t)", "Vitesse v(t)"])
    input_val = st.number_input("Entrer la valeur", value=0.0, step=0.1, key=f"inverse_{sim_id}")

    if option == "Position x(t)":
        coeffs = [a2, b2, c2 - input_val]
        t_solutions = np.roots(coeffs)
        t_solutions = t_solutions[np.isreal(t_solutions)].real
        st.write(f"Temps possibles : {t_solutions}")
        st.markdown("Formule quadratique appliquée :")
        st.latex(rf"t = \frac{{ -({b2:.3f}) \pm \sqrt{{({b2:.3f})^2 - 4 ({a2:.3f}) ({c2 - input_val:.3f})}} }}{{ 2 ({a2:.3f}) }}")

    elif option == "Vitesse v(t)":
        t_sol = (input_val - b2) / (2*a2)
        st.write(f"Temps : {t_sol:.3f} s")
        st.markdown("Formule appliquée :")
        st.latex(rf"t = \frac{{ {input_val:.3f} - ({b2:.3f}) }}{{ 2 ({a2:.3f}) }}")

    # =======================
    # 8️⃣ Modifier / Supprimer simulation
    # =======================
    st.header("8️⃣ Modifier / Supprimer cette simulation")
    edited_df = st.data_editor(df, num_rows="dynamic", key=f"editor_{sim_id}")

    if st.button("Sauvegarder modifications", key=f"save_{sim_id}"):
        updated_data = edited_df.to_dict(orient="list")
        supabase.table("cinematique_brute").update({"results": updated_data}).eq("id", sim_id).execute()
        st.success("Simulation mise à jour ✅")
        st.experimental_rerun()

    if st.button("Supprimer simulation", key=f"delete_{sim_id}"):
        supabase.table("cinematique_brute").delete().eq("id", sim_id).execute()
        st.success("Simulation supprimée ✅")
        st.experimental_rerun()
