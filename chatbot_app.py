import streamlit as st
from chatbot_module import respond_to_user

# Titre de l'application
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("logo.jpeg", width=100)
with col2:
    st.title("Chatbot Ravenfox")

# Champ de saisie pour la question de l'utilisateur
user_query = st.text_input("Posez votre question:")

# Bouton pour soumettre la question
if st.button("Envoyer"):
    if user_query:
        # Obtenir la réponse du bot
        response = respond_to_user(user_query)
        
        # Afficher la réponse
        st.text_area("Réponse du bot:", value=response, height=200)
    else:
        st.warning("Veuillez entrer une question.")