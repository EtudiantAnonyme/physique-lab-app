import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.supabase_client import supabase

def run_cinematique_lab():
    st.set_page_config(
        page_title="Laboratoire Cinématique 1D",
        page_icon="🧪",
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
    # 1️⃣ AJOUT DE DONNÉES BRUTES
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
    # 2️⃣ RÉCUPÉRATION DES DONNÉES
    # =======================
    st.header("2️⃣ Gestion des simulations")
    response = supabase.table("cinematique_brute").select("*").execute()
    simulations = response.data

    if not simulations:
        st.info("Aucune simulation enregistrée.")
        return

    # =======================
    # 3️⃣ ANALYSE D'UNE SIMULATION
    # =======================
    for sim in simulations:
        sim_id = sim["id"]
        st.markdown(f"## Simulation {sim_id} — {sim['created_at']}")

        df = pd.DataFrame(sim["results"])
        df = df.sort_values("t")
        t_vals = df["t"].values
        x_vals = df["x"].values

        # ---- Ajustement Linéaire ----
        a1, b1 = np.polyfit(t_vals, x_vals, 1)
        x_lin_fit = a1 * t_vals + b1
        r2_lin = 1 - np.sum((x_vals - x_lin_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)

        # ---- Ajustement Quadratique ----
        a2, b2, c2 = np.polyfit(t_vals, x_vals, 2)
        x_quad_fit = a2 * t_vals**2 + b2 * t_vals + c2
        r2_quad = 1 - np.sum((x_vals - x_quad_fit)**2) / np.sum((x_vals - np.mean(x_vals))**2)

        # ---- Graphique moderne ----
        t_smooth = np.linspace(t_vals.min(), t_vals.max(), 300)
        x_lin_smooth = a1 * t_smooth + b1
        x_quad_smooth = a2 * t_smooth**2 + b2 * t_smooth + c2

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t_vals, x_vals, s=25, color="#1f2937", label="Données expérimentales")
        ax.plot(t_smooth, x_lin_smooth, linewidth=2, color="royalblue", label="Approximation linéaire")
        ax.plot(t_smooth, x_quad_smooth, linewidth=2.5, color="crimson", linestyle="--", label="Approximation quadratique")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Position en fonction du temps")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        st.pyplot(fig)

        # ---- Formules et dérivées ----
        st.subheader("3️⃣ Modèle mathématique et dérivées")
        st.markdown("**Fonction quadratique ajustée :**")
        st.latex(rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}")
        st.markdown("**Vitesse et accélération calculées à partir du modèle :**")
        st.latex(r"v(t) = \frac{dx}{dt} = 2 a t + b")
        st.latex(rf"v(t) = {2*a2:.3f} t + {b2:.3f}")
        st.latex(r"a(t) = \frac{d^2x}{dt^2} = 2 a")
        st.latex(rf"a(t) = {2*a2:.3f}")

        # ---- Coefficient R² ----
        st.markdown("**Qualité de l’ajustement :**")
        st.write(f"Linéaire : R² = {r2_lin:.4f}")
        st.write(f"Quadratique : R² = {r2_quad:.4f}")

        # =======================
        # 4️⃣ Tableau cinématique
        # =======================
        st.subheader("4️⃣ Tableau cinématique")
        v_model = 2 * a2 * t_vals + b2
        a_model = np.full_like(t_vals, 2 * a2)
        df_phys = pd.DataFrame({
            "t (s)": t_vals,
            "x(t) (m)": x_vals,
            "v(t) (m/s)": v_model,
            "a(t) (m/s²)": a_model
        })
        st.dataframe(df_phys, use_container_width=True)
        st.subheader("4️⃣ Tableau cinématique")
        v_model = 2 * a2 * t_vals + b2
        a_model = np.full_like(t_vals, 2 * a2)
        df_phys = pd.DataFrame({
            "t (s)": t_vals,
            "x(t) (m)": x_vals,
            "v(t) (m/s)": v_model,
            "a(t) (m/s²)": a_model
        })
        st.dataframe(df_phys, use_container_width=True)

# Ajouter des zones copiables pour chaque colonne
st.markdown("**📋 Copier les données d'une colonne :**")
for col in df_phys.columns:
    col_data = ", ".join(df_phys[col].astype(str))
    st.text_area(f"Copier la colonne '{col}'", value=col_data, height=80)
        # =======================
        # 5️⃣ Calculatrice cinématique
        # =======================
        st.subheader("5️⃣ Calculatrice interactive")

        st.markdown("Vous pouvez calculer x, v, a pour un temps donné, même en dehors des mesures expérimentales.")
        t_input = st.number_input("Entrer un temps t (s)", value=float(t_vals[-1]), step=0.1, key=f"t_calc_{sim_id}")
        x_calc = a2 * t_input**2 + b2 * t_input + c2
        v_calc = 2 * a2 * t_input + b2
        a_calc = 2 * a2
        col1, col2, col3 = st.columns(3)
        col1.metric("Position x(t)", f"{x_calc:.3f} m")
        col2.metric("Vitesse v(t)", f"{v_calc:.3f} m/s")
        col3.metric("Accélération a(t)", f"{a_calc:.3f} m/s²")

        st.markdown("**Explication des calculs :**")
        # Formule générale avec coefficients numériques
        st.markdown("**Formules appliquées avec les coefficients du modèle quadratique :**")
        st.latex(rf"x(t) = {a2:.3f} t^2 + {b2:.3f} t + {c2:.3f}")
        st.latex(rf"v(t) = dx/dt = 2 \cdot {a2:.3f} t + {b2:.3f}")
        st.latex(rf"a(t) = d^2x/dt^2 = 2 \cdot {a2:.3f}")

        # Formule appliquée à un temps spécifique
        st.markdown(f"**Substitution pour t = {t_input} s :**")
        st.latex(rf"x({t_input}) = {a2:.3f} \cdot ({t_input})^2 + {b2:.3f} \cdot ({t_input}) + {c2:.3f} = {x_calc:.3f}")
        st.latex(rf"v({t_input}) = 2 \cdot {a2:.3f} \cdot ({t_input}) + {b2:.3f} = {v_calc:.3f}")
        st.latex(rf"a({t_input}) = 2 \cdot {a2:.3f} = {a_calc:.3f}")


        # =======================
        # 6️⃣ Inverser : calculer t à partir de x ou v
        # =======================
        st.subheader("6️⃣ Calculer le temps à partir de x ou v")

        option = st.selectbox("Choisir la variable connue", ["Position x(t)", "Vitesse v(t)"])
        input_val = st.number_input("Entrer la valeur", value=0.0, step=0.1, key=f"inverse_{sim_id}")

        if option == "Position x(t)":
            # Résolution quadratique : a t^2 + b t + c - x = 0
            coeffs = [a2, b2, c2 - input_val]
            t_solutions = np.roots(coeffs)
            t_solutions = t_solutions[np.isreal(t_solutions)].real
            st.write(f"Temps possibles : {t_solutions}")

            st.markdown("**Formule quadratique appliquée :**")

            # Écriture explicite de la solution quadratique
            # t = (-b ± sqrt(b^2 - 4 a (c - x))) / (2a)
            a_str = f"{a2:.3f}"
            b_str = f"{b2:.3f}"
            c_minus_x_str = f"{c2 - input_val:.3f}"

            latex_eq = (
                rf"t = \frac{{ -({b_str}) \pm \sqrt{{({b_str})^2 - 4 \cdot ({a_str}) \cdot ({c_minus_x_str})}} }}"
                rf"{{2 \cdot ({a_str})}}"
            )

            st.latex(latex_eq)


        elif option == "Vitesse v(t)":
            # Résolution linéaire : 2 a t + b - v = 0
            t_sol = (input_val - b2) / (2 * a2)
            st.write(f"Temps : {t_sol:.3f} s")
            st.markdown("**Formule appliquée :**")
            a_str = f"{a2:.3f}"
            b_str = f"{b2:.3f}"
            v_val_str = f"{input_val:.3f}"

            # Formule explicite résolue pour t : t = (v - b) / (2 a)
            latex_t = rf"t = \frac{{ {v_val_str} - ({b_str}) }}{{ 2 \cdot ({a_str}) }}"

            st.latex(latex_t)


        # =======================
        # 7️⃣ Modifier / Supprimer
        # =======================
        st.subheader("7️⃣ Modifier / Supprimer cette simulation")
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
