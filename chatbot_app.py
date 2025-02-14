import importlib
import streamlit as st
import chatbot_module
importlib.reload(chatbot_module)
from chatbot_module import respond_to_user

# Titre de l'application avec logo au-dessus
st.image("logo.jpeg", width=100)
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