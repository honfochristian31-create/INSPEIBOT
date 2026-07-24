import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq

# Charger la clé API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Charger les données et créer l'index (une seule fois, au démarrage)
@st.cache_resource
def charger_systeme():
    with open('data/inspei.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    textes = [f"Q: {item['question']} R: {item['answer']}" for item in data]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vecteurs = model.encode(textes, show_progress_bar=False)
    vecteurs = np.array(vecteurs).astype('float32')
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    
    return data, model, index

def rechercher(question, model, index, data, k=3):
    vecteur_question = model.encode([question])
    vecteur_question = np.array(vecteur_question).astype('float32')
    distances, indices = index.search(vecteur_question, k)
    
    resultats = []
    for i, idx in enumerate(indices[0]):
        if idx < len(data):
            similarite = 1 / (1 + distances[0][i])
            resultats.append({
                'question': data[idx]['question'],
                'reponse': data[idx]['answer'],
                'similarite': float(similarite)
            })
    return resultats

def repondre(question, model, index, data):
    resultats = rechercher(question, model, index, data, k=3)
    
    if resultats and resultats[0]['similarite'] > 0.70:
        return resultats[0]['reponse']
    
    contexte = ""
    for i, res in enumerate(resultats):
        contexte += f"Document {i+1}:\nQuestion: {res['question']}\nRéponse: {res['reponse']}\n\n"
    
    messages = [
        {"role": "system", "content": f"""Tu es un assistant expert de l'INSPEI (Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur) au Bénin.

Voici des informations disponibles sur l'INSPEI :
{contexte}

RÈGLES :
- Réponds uniquement en utilisant les informations du contexte.
- Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information dans ma base. Contacte le secrétariat de l'INSPEI pour plus de détails."
- Sois clair, précis, et adapté à des étudiants béninois.
- Utilise des emojis 🎓📚⚙️🇧🇯 pour rendre le ton dynamique."""},
        {"role": "user", "content": question}
    ]
    
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return reponse.choices[0].message.content

# ---------- INTERFACE STREAMLIT ----------
st.set_page_config(page_title="Chatbot INSPEI", page_icon="🎓")
st.title("🎓 Assistant INSPEI")
st.markdown("Posez vos questions sur l'INSPEI (admission, campus, filières, etc.)")

# Charger le système
data, model, index = charger_systeme()

# Gérer l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            reponse = repondre(prompt, model, index, data)
            st.markdown(reponse)
    
    st.session_state.messages.append({"role": "assistant", "content": reponse})