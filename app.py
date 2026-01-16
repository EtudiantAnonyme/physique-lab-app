import streamlit as st
from labs.cinematique import run_cinematique_lab

st.set_page_config(
    page_title="Application laboratoire en physique",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Site web pour laboratoires en physique - Montmorency")
st.markdown("""
Bienvenue sur le site web dédié aux outils de laboratoires en physique du Cégep Montmorency.  
Ce site vise à fournir des ressources et des applications interactives pour faciliter les expériences de laboratoire en physique.
""")

st.sidebar.header("Navigation")
pages = ["Accueil", "Cinématique 1D"]
page_choice = st.sidebar.selectbox("Choisissez une page", pages)

if page_choice == "Accueil":
    st.header("Introduction")
    st.write("""
    Cette application est un prototype pour les laboratoires en physique.  
    Elle permettra à terme de :
    - Lancer des simulations interactives
    - Visualiser des graphiques de position, vitesse et accélération
    - Explorer différents phénomènes physiques
    """)
    st.info("Pour l'instant, seule la page d'accueil est disponible. Les expériences seront ajoutées prochainement.")
elif page_choice == "Cinématique 1D":
    run_cinematique_lab()
