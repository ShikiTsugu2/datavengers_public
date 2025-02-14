#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import fitz  # PyMuPDF
import pandas as pd

# Dossier contenant les fichiers PDF
pdf_folder = 'pdf'

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Liste des fichiers PDF dans le dossier
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Initialisation des listes pour le dataframe
texts = []
labels = []

# Parcours des fichiers PDF et extraction du texte et des labels
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    texts.append(text)
    
    if 'BC' in pdf_file:
        labels.append('BC')
    elif 'Brochure' in pdf_file:
        labels.append('Brochure')
    elif 'Manuel d\'utilisation' in pdf_file or 'MU' in pdf_file:
        labels.append('MU')
    elif 'MM' in pdf_file:
        labels.append('MM')
    else:
        labels.append('Unknown')

# Création du dataframe
df = pd.DataFrame({'Text': texts, 'Label': labels})


# In[2]:


df


# In[3]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy
import re

# Télécharger les ressources nécessaires de nltk
'''
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
'''

# Charger le modèle de langue de spacy
try:
    nlp = spacy.load('fr_core_news_sm')
except OSError:
    from spacy.cli import download
    download('fr_core_news_sm')
    nlp = spacy.load('fr_core_news_sm')

# Initialiser les objets nécessaires
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    # Suppression des caractères spéciaux et des sauts de ligne
    text = re.sub(r'\W+', ' ', text)
    text = text.replace('\n', ' ')
    
    # Tokenisation
    tokens = nltk.word_tokenize(text)
    
    # Normalisation et suppression des stop words
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    # Lemmatisation
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# Appliquer le traitement de texte à chaque texte dans la liste
processed_texts = [preprocess_text(text) for text in texts]

# Mettre à jour le dataframe avec les textes traités
df['Processed_Text'] = processed_texts
df


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer()

# Vectoriser les textes prétraités
tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])

def search_documents(query, top_n=5):
    # Prétraiter la requête de l'utilisateur
    processed_query = preprocess_text(query)
    
    # Vectoriser la requête
    query_vector = vectorizer.transform([processed_query])
    
    # Calculer la similarité de cosinus entre la requête et les documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Obtenir les indices des documents les plus similaires
    related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Retourner les documents les plus similaires
    return df.iloc[related_docs_indices]

# Exemple de recherche
query = "manuel d'utilisation fauteuil roulant invacare avivia rx"
result = search_documents(query)
result


# In[5]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import requests
import json

# Étape 1: Analyse de la requête
def analyze_query(query):
    processed_query = preprocess_text(query)
    return processed_query

# Étape 2: Classification du problème
# Entraîner un classificateur pour la classification du problème
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier.fit(df['Processed_Text'], df['Label'])

def classify_problem(query):
    processed_query = analyze_query(query)
    label = classifier.predict([processed_query])[0]
    return label

# Étape 3: Base de règles
def apply_rules(label):
    rules = {
        'BC': 'Règle pour BC',
        'Brochure': 'Règle pour Brochure',
        'MU': 'Règle pour Manuel d\'utilisation',
        'MM': 'Règle pour MM',
        'Unknown': 'Règle pour inconnu'
    }
    return rules.get(label, 'Règle par défaut')

# Étape 4: Recherche dans les PDF
def search_pdfs(query):
    result = search_documents(query)
    return result

# Étape 5: Filtrage et sélection de l'information
def filter_information(result):
    filtered_info = result['Text'].tolist()
    return filtered_info

# Étape 6: Génération d'une réponse détaillée => à modifier pour utiliser une IA générative et fournir une réponse plus précise
def generate_response(query, filtered_info):
    prompt = "Voici la question : " + query + "Voici les informations trouvées par rapport à la question :\n\n"
    for info in filtered_info:
        prompt += info + "\n\n"
    
    url = "http://localhost:11434/api/generate"

    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    actual_response = response.text
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data['response']
    
    return actual_response

# Étape 7: Réponse à l'utilisateur
def respond_to_user(query):
    label = classify_problem(query)
    rule = apply_rules(label)
    result = search_pdfs(query)
    filtered_info = filter_information(result)
    response = generate_response(query, filtered_info)
    return response


# In[6]:


# Exemple d'utilisation
query = "mon fauteuil roulant invacare avivia rx est trop bas, comment le réajuster ?"
response = respond_to_user(query)
print(response)


# In[ ]:


#

