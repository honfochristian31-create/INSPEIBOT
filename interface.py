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
    """Charge le fichier JSON et crée l'index FAISS"""
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
    """Recherche les k questions/réponses les plus proches"""
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

# ---------- 5. RÉPONSE PRINCIPALE ----------
def repondre(question, model, index, data, historique=[]):
    """
    Génère une réponse en utilisant :
    - Le JSON si la similarité est > 60%
    - Groq avec historique et contexte sinon
    """
    # 0. Gestion des salutations
    if est_salutation(question) and len(historique) == 0:
        return reponse_salutation()
    
    # 1. Recherche dans la base de connaissances
    resultats = rechercher(question, model, index, data, k=3)
    
    # 2. Si similarité > 60%, réponse directe du JSON
    if resultats and resultats[0]['similarite'] > 0.60:
        return resultats[0]['reponse']
    
    # 3. Sinon, préparation du contexte pour Groq
    contexte = ""
    
    # 3a. Ajouter l'historique récent
    if historique:
        contexte += "📜 HISTORIQUE DE LA CONVERSATION :\n"
        for msg in historique[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            contexte += f"{role} : {msg['content']}\n"
        contexte += "\n"
    
    # 3b. Ajouter les résultats de la recherche
    if resultats:
        contexte += "📚 INFORMATIONS DISPONIBLES :\n"
        for i, res in enumerate(resultats[:3]):
            contexte += f"Document {i+1} :\n"
            contexte += f"Question : {res['question']}\n"
            contexte += f"Réponse : {res['reponse']}\n\n"
    else:
        contexte += "📚 Aucune information trouvée dans la base.\n"
    
    # 4. Appel à Groq avec le contexte
    messages = [
        {"role": "system", "content": f"""Tu es un assistant expert de l'INSPEI (Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur) au Bénin.

CONTEXTE DISPONIBLE :
{contexte}

RÈGLES CRITIQUES (À SUIVRE ABSOLUMENT) :
1. Utilise UNIQUEMENT les informations du contexte pour répondre.
2. Si l'information n'est pas dans le contexte, réponds EXACTEMENT : 
   "Je ne trouve pas cette information dans ma base. Consultez le site officiel https://siteinspei.netlify.app ou contactez l'INSPEI."
3. **NE DIS PAS "Bonjour" ou "Salut"** à chaque réponse. Continue naturellement la conversation.
4. Si c'est une question de suivi (ex: "Et les débouchés ?", "Explique mieux"), réfère-toi à l'historique.
5. Sois clair, précis, et adapté à des étudiants béninois.
6. Utilise un ton professionnel mais bienveillant. N'invente rien."""},
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
        
        # 5. VÉRIFICATION POST-GÉNÉRATION (fallback supplémentaire)
        mots_suspects = ["je ne sais pas", "désolé", "je n'ai pas trouvé", "je ne trouve pas", "aucune information"]
        if any(mot in reponse_texte.lower() for mot in mots_suspects):
            return f"{reponse_texte}\n\n💡 Pour plus d'informations : https://siteinspei.netlify.app"
        
        if len(reponse_texte) < 30:
            return "Je n'ai pas assez d'informations pour répondre. Consultez le site officiel https://siteinspei.netlify.app ou posez une question plus précise."
        
        return reponse_texte
        
    except Exception as e:
        # 6. Fallback ultime (si Groq plante)
        return f"❌ Une erreur est survenue. Veuillez réessayer ou consulter le site officiel : https://siteinspei.netlify.app"

# ---------- 6. INTERFACE STREAMLIT ----------
st.set_page_config(
    page_title="Assistant INSPEI",
    page_icon="🎓",
    layout="centered"
)

# En-tête personnalisé avec CSS
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
""", unsafe_allow_html=True)

# Titre personnalisé
st.markdown("""
<div class="main-header">
    <h1>🎓 Assistant INSPEI</h1>
    <p>Votre guide pour l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur</p>
</div>
""", unsafe_allow_html=True)

# Charger le système
data, model, index = charger_systeme()

# Vérifier que tout est chargé
if not data:
    st.warning("⚠️ Aucune donnée chargée. Vérifiez que le fichier data/inspei.json existe.")
    st.stop()

# Initialiser l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- 7. GESTION DE L'INPUT UTILISATEUR ----------
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Obtenir la réponse avec historique
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            historique = st.session_state.messages[:-1] if st.session_state.messages else []
            reponse = repondre(prompt, model, index, data, historique)
            st.markdown(reponse)
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse})

# ---------- 8. BOUTON "NOUVELLE CONVERSATION" ----------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Nouvelle conversation"):
    st.session_state.messages = []
    st.rerun()

# Pied de page discret
st.markdown("""
<div class="footer">
    INSPEI &bull; Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur
</div>
""", unsafe_allow_html=True)
