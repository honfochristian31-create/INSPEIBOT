import streamlit as st
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq

# ---------- 1. CONFIGURATION ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- 2. CHARGEMENT DU SYSTÈME ----------
@st.cache_resource
def charger_systeme():
    try:
        with open('data/inspei.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("❌ Fichier data/inspei.json introuvable !")
        return [], None, None
    
    textes = [f"Q: {item['question']} R: {item['answer']}" for item in data]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vecteurs = model.encode(textes, show_progress_bar=False)
    vecteurs = np.array(vecteurs).astype('float32')
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    
    return data, model, index

# ---------- 3. RECHERCHE DANS LE JSON ----------
def rechercher(question, model, index, data, k=3):
    if len(data) == 0 or model is None:
        return []
    
    try:
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
    except Exception as e:
        return []

# ---------- 4. SALUTATIONS ----------
SALUTATIONS = ["bonjour", "salut", "cc", "coucou", "hello", "hi", "yo", "bonsoir", "slt"]

def est_salutation(question):
    return question.lower().strip() in SALUTATIONS

def reponse_salutation():
    responses = [
        "Bonjour ! 😊 Posez-moi vos questions sur l'INSPEI, les admissions, les filières, les écoles ou la vie étudiante.",
        "Salut ! 👋 Comment puis-je vous aider ? Je suis là pour vous renseigner sur l'INSPEI et les classes préparatoires.",
        "Bonjour et bienvenue ! 🎓 Que souhaitez-vous savoir sur l'INSPEI ?"
    ]
    return random.choice(responses)

# ---------- 5. FONCTION PRINCIPALE ----------
def repondre(question, model, index, data, historique=[]):
    # 0. Salutations
    if est_salutation(question) and len(historique) == 0:
        return reponse_salutation()
    
    # 1. Détecter les questions très courtes sur la localisation
    mots_localisation = ["ou", "où", "localisation", "adresse", "situé", "trouve"]
    if len(question.split()) <= 4 and any(mot in question.lower() for mot in mots_localisation):
        # Forcer la recherche de la question "Où se trouve l'INSPEI ?"
        resultats_force = rechercher("Où se trouve l'INSPEI ?", model, index, data, k=1)
        if resultats_force and resultats_force[0]['similarite'] > 0.50:
            return resultats_force[0]['reponse']
    
    # 2. Recherche normale
    resultats = rechercher(question, model, index, data, k=3)
    
    # 3. Si similarité > 60% ou > 50% pour les questions courtes
    seuil = 0.60
    if len(question.split()) <= 3:
        seuil = 0.45  # seuil plus bas pour les questions très courtes
    
    if resultats and resultats[0]['similarite'] > seuil:
        return resultats[0]['reponse']
    
    # 4. Préparation du contexte pour Groq
    contexte = ""
    
    if historique:
        contexte += "📜 CONVERSATION :\n"
        for msg in historique[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            contexte += f"{role} : {msg['content']}\n"
        contexte += "\n"
    
    if resultats:
        contexte += "📚 INFORMATIONS DISPONIBLES :\n"
        for i, res in enumerate(resultats[:3]):
            contexte += f"Document {i+1} :\n"
            contexte += f"Question : {res['question']}\n"
            contexte += f"Réponse : {res['reponse']}\n\n"
    else:
        contexte += "📚 Aucune information disponible.\n"
    
    # 5. Prompt système (naturel, sans jargon)
    messages = [
        {"role": "system", "content": f"""Tu es un conseiller pédagogique expert de l'INSPEI (Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur) au Bénin.

Voici les informations dont tu disposes :
{contexte}

RÈGLES DE RÉPONSE (À SUIVRE ABSOLUMENT) :

1. **Ton identité** : Tu es un conseiller pédagogique expérimenté. Tu parles de façon naturelle, comme un humain.

2. **Si tu connais la réponse** : Réponds de manière claire, précise et bienveillante, en t'appuyant sur les informations ci-dessus.

3. **Si tu ne connais pas la réponse** : 
   - Ne dis JAMAIS : "je ne trouve pas dans ma base", "selon les documents", "d'après les données", "historique", "fichiers", "entraînement".
   - Dis plutôt naturellement : "Je ne dispose pas de cette information précise pour le moment. Je vous conseille de consulter le site officiel de l'INSPEI ou de contacter directement le secrétariat pour obtenir des détails."

4. **Interdictions formelles** :
   - Ne mentionne JAMAIS que tu utilises une "base de données", des "documents", un "historique" ou une "mémoire".
   - Ne dis pas "Bonjour" ou "Salut" à chaque message. Continue naturellement la conversation.
   - Sois concis, va à l'essentiel, mais reste professionnel.

5. **Questions de suivi** : Réponds en faisant le lien avec ce que tu as dit précédemment, mais sans jamais dire "comme je l'ai mentionné dans l'historique". Dis simplement "Comme je vous l'indiquais..." ou "Pour compléter..."."""},
        {"role": "user", "content": question}
    ]
    
    try:
        reponse = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )
        reponse_texte = reponse.choices[0].message.content
        
        # Nettoyage des termes techniques
        termes_techniques = ["base de données", "documents", "historique", "fichiers", "entraînement", "data", "corpus"]
        for terme in termes_techniques:
            if terme in reponse_texte.lower():
                reponse_texte = reponse_texte.replace(terme, "à ma connaissance")
                reponse_texte = reponse_texte.replace("dans ma base", "pour le moment")
                reponse_texte = reponse_texte.replace("selon les documents", "")
                reponse_texte = reponse_texte.replace("qui m'ont été fournis", "")
                reponse_texte = reponse_texte.replace("https://siteinspei.netlify.app", "le site officiel")
        
        if len(reponse_texte) < 30:
            return "Je n'ai pas assez d'informations pour répondre à cette question. Je vous conseille de consulter le site officiel de l'INSPEI ou de contacter directement le secrétariat."
        
        return reponse_texte
        
    except Exception as e:
        return "Désolé, une erreur s'est produite. Veuillez réessayer ou consulter le site officiel de l'INSPEI."

# ---------- 6. INTERFACE STREAMLIT ----------
st.set_page_config(
    page_title="Assistant INSPEI",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #1a3a5c;
        margin-bottom: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
    }
    .footer {
        text-align: center;
        padding: 1rem 0;
        font-size: 0.85rem;
        color: #888;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
<div class="main-header">
    <h1>🎓 Assistant INSPEI</h1>
    <p>Votre guide pour l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur</p>
</div>
""", unsafe_allow_html=True)

data, model, index = charger_systeme()

if not data:
    st.warning("⚠️ Aucune donnée chargée. Vérifiez que le fichier data/inspei.json existe.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            historique = st.session_state.messages[:-1] if st.session_state.messages else []
            reponse = repondre(prompt, model, index, data, historique)
            st.markdown(reponse)
    
    st.session_state.messages.append({"role": "assistant", "content": reponse})

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Nouvelle conversation"):
    st.session_state.messages = []
    st.rerun()

st.markdown("""
<div class="footer">
    INSPEI &bull; Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur
</div>
""", unsafe_allow_html=True)
