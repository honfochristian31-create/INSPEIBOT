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

# ---------- 3. RECHERCHE ----------
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

# ---------- 5. RÉPONSE PRINCIPALE ----------
def repondre(question, model, index, data, historique=[]):
    # 0. Salutations
    if est_salutation(question) and len(historique) == 0:
        return reponse_salutation()
    
    question_lower = question.lower().strip()
    
    # --- RÈGLE : "repete" ou "répète" ---
    if question_lower in ["repete", "répète", "repetes", "répètes", "repeter", "répéter"]:
        return "Je suis à votre disposition pour toute question sur l'INSPEI. Que souhaitez-vous savoir ?"
    
    # --- RÈGLE : confirmations (OK, OUI, NON, MERCI, etc.) ---
    mots_confirmation = ["ok", "oui", "non", "merci", "d'accord", "super", "parfait", "cool", "okay", "yes", "no", "si", "sisi"]
    if question_lower.strip() in mots_confirmation:
        if question_lower in ["ok", "d'accord", "super", "parfait", "cool", "okay"]:
            return "Parfait ! 😊 N'hésitez pas si vous avez d'autres questions sur l'INSPEI, les admissions, les filières ou les écoles d'ingénieurs."
        elif question_lower in ["merci", "thanks"]:
            return "Je vous en prie ! 😊 C'est un plaisir de vous aider. Revenez quand vous voulez."
        elif question_lower in ["oui", "yes", "si", "sisi"]:
            return "Excellent ! Que voulez-vous savoir d'autre ?"
        elif question_lower in ["non", "no"]:
            return "Pas de problème. Si vous avez d'autres questions, je suis là pour vous aider."
        else:
            return "Je suis à votre disposition pour toute question sur l'INSPEI."
    
    # --- RÈGLE : "COMMENT FAIRE" (demande de méthode) ---
    mots_comment = ["comment faire", "comment je fais", "comment puis-je", "que dois-je", "je fais comment", "comment procéder"]
    if any(mot in question_lower for mot in mots_comment):
        if "info" in question_lower or "plus" in question_lower:
            return "Pour obtenir plus d'informations, il vous suffit de poser une question précise sur l'un des sujets suivants :\n\n• Tapez 'Matières' pour connaître les matières par semestre\n• Tapez 'Écoles' pour en savoir plus sur l'ENSGEP, ENSGMM, ENSTP\n• Tapez 'Admission' pour les conditions d'admission et le concours\n• Tapez 'Vie étudiante' pour l'hébergement et la restauration\n• Tapez 'Localisation' pour savoir où se trouve l'INSPEI\n• Tapez 'Administration' pour les responsables\n• Tapez 'Frais' pour les frais et bourses\n• Tapez 'Événements' pour les sorties pédagogiques\n\nPosez votre question simplement, je suis là pour vous aider ! 😊"
    
    # --- RÈGLE : "PLUS D INFOS" (demande de précision) ---
    mots_plus_infos = ["plus d'infos", "plus d info", "plus d'informations", "plus dinfos", "en savoir plus", "plus de details", "plus de détails"]
    if question_lower in mots_plus_infos or ("plus" in question_lower and ("info" in question_lower or "detail" in question_lower)):
        return "Je peux vous donner plus d'informations sur un sujet spécifique. Dites-moi ce qui vous intéresse :\n\n• 📚 Les **matières** enseignées par semestre\n• 🎓 Les **écoles d'ingénieurs** (ENSGEP, ENSGMM, ENSTP)\n• 📝 Les **conditions d'admission** et le concours\n• 🏠 La **vie étudiante** (hébergement, restauration)\n• 📍 La **localisation** de l'INSPEI\n• 👨‍🏫 L'**administration** et les responsables\n• 💰 Les **frais et bourses**\n• 📅 Les **événements** et sorties pédagogiques\n\nPosez-moi votre question précise et je vous répondrai ! 😊"
    
    # --- RÈGLE PRIORITAIRE : "C'EST QUOI" (définition) ---
    if ("quoi" in question_lower or "definition" in question_lower or "c'est quoi" in question_lower or "qu'est-ce" in question_lower) and "inspei" in question_lower:
        return "L'INSPEI est l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur. C'est un établissement public rattaché à l'UNSTIM (Université Nationale des Sciences, Technologies, Ingénierie et Mathématiques). Il a été officiellement créé par l'arrêté N°719/MESRS/... du 23/12/2020, mais a démarré ses activités dès 2016-2017. Sa mission est de former des bacheliers scientifiques pour les grandes écoles d'ingénieurs du Bénin. La formation dure deux ans et débouche sur le CPEI."
    
    # --- RÈGLE POUR LA LOCALISATION ---
    if "inspei" in question_lower:
        mots_localisation = ["ou", "où", "u", "est", "trouve", "situé", "localisation", "adresse", "emplacement", "cote"]
        for mot in mots_localisation:
            if mot in question_lower:
                return "L'INSPEI est situé en République du Bénin, dans le Département du Zou, à Abomey, à environ 1 km de la place Goho, sur la route RNIE2 en allant vers Bohicon, à Sogbo-Aliho."
    
    # --- RÈGLE POUR LES ADMINISTRATEURS ---
    mots_admin = ["administrateur", "admin", "direction", "responsable", "membres", "dirigeant", "patron"]
    if any(mot in question_lower for mot in mots_admin):
        return "L'équipe dirigeante et administrative de l'INSPEI comprend : Dr (MC) AKOWANOU Christian D. (Directeur), Dr. Bernard N. TOKPOHOZIN (Chef du Service de la Scolarité et des Examens), GBEGNITO Wilfried Hodonou (Secrétaire général), le Comptable, le Chef matériel, et AKPAVOU Chédrac (Conducteur de Bus)."
    
    # --- RÈGLE POUR LES MATIÈRES ---
    if question_lower in ["les matieres", "matieres", "les matière", "matière"]:
        return "Les matières enseignées à l'INSPEI sont réparties sur 4 semestres. Voici le programme :\n\nSemestre 1 : Algorithmique, Thermodynamique, Maths 1, Chimie de l'Ingénieur, EPS, TEMC, Probabilités et Statistiques, Statique Graphique et Analytique.\n\nSemestre 2 : Analyse Numérique, Graphe et Optimisation, Maths 2, Cinématique et Dynamique, Langage (C/Python), RDM, Normes et Mesures, Anglais technique.\n\nSemestre 3 : TEMC, Recherche Opérationnelle, Mécanique des Fluides, Maths 3, Physique des Matériaux, Géométrie Descriptive, Dessin Technique et DAO, Électricité Générale.\n\nSemestre 4 : Maths 4, Matlab, MPA, Sciences Biologiques pour l'Ingénieur, Transfert Thermique, Ondes Électromagnétiques, Anglais Technique Avancé, EPS."
    
    # --- RÈGLE POUR "COMPOS" (épreuves du concours) ---
    mots_compos = ["compos", "composition", "épreuve", "compose", "epreuve", "concours écrit"]
    if any(mot in question_lower for mot in mots_compos):
        return "Les épreuves du concours d'entrée à l'INSPEI se déroulent généralement en centres d'examen : Abomey (ENSTP/UNSTIM), Cotonou (CEG Gbégamey, Collège Catholique Notre Dame des Apôtres, CEG Ste Rita, CEG les Pylônes) et Parakou (IFSIO). Les matières évaluées sont les Mathématiques, la Physique, la Chimie et la Technologie. Consultez l'avis de concours officiel pour les détails précis de l'année en cours sur www.concours.enseignementsuperieur.gouv.bj."
    
    # --- Si la question est trop courte ---
    if len(question.strip().split()) <= 2:
        return "Pouvez-vous préciser votre question sur l'INSPEI ? Je suis là pour vous renseigner sur les admissions, les filières, les écoles, la vie étudiante, etc."
    
    # 2. Recherche normale dans le JSON
    resultats = rechercher(question, model, index, data, k=3)
    
    # 3. Seuil adapté
    seuil = 0.60
    if len(question.split()) <= 3:
        seuil = 0.45
    
    if resultats and resultats[0]['similarite'] > seuil:
        return resultats[0]['reponse']
    
    # 4. Préparation du contexte pour Groq
    contexte = ""
    
    if historique:
        for msg in historique[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            contexte += f"{role} : {msg['content']}\n"
        contexte += "\n"
    
    if resultats:
        for i, res in enumerate(resultats[:3]):
            contexte += f"Question : {res['question']}\n"
            contexte += f"Réponse : {res['reponse']}\n\n"
    else:
        contexte += "Aucune information disponible.\n"
    
    # 5. Prompt système
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
   - Ne répète PAS le contexte ou l'historique dans ta réponse.
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
        termes_techniques = ["base de données", "documents", "historique", "fichiers", "entraînement", "data", "corpus", "contexte"]
        for terme in termes_techniques:
            if terme in reponse_texte.lower():
                reponse_texte = reponse_texte.replace(terme, "")
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
    layout="wide"
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

# ---------- 7. GESTION DES CONVERSATIONS ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = []
    st.session_state.current_convo_id = None

def nouvelle_conversation(titre="INSPEI - Nouvelle conversation"):
    convo_id = len(st.session_state.conversations) + 1
    st.session_state.conversations.append({
        "id": convo_id,
        "titre": titre,
        "messages": [],
        "created_at": "Maintenant"
    })
    st.session_state.current_convo_id = convo_id
    return convo_id

def charger_conversation(convo_id):
    st.session_state.current_convo_id = convo_id

def get_active_conversation():
    for convo in st.session_state.conversations:
        if convo["id"] == st.session_state.current_convo_id:
            return convo
    return None

def generer_titre(question):
    question_lower = question.lower()
    mots_cles = {
        "admission": "Admission",
        "concours": "Concours",
        "inscription": "Inscription",
        "filière": "Filières",
        "filiere": "Filières",
        "école": "Écoles",
        "ecole": "Écoles",
        "ensgep": "ENSGEP",
        "ensgmm": "ENSGMM",
        "enstp": "ENSTP",
        "matière": "Matières",
        "matiere": "Matières",
        "semestre": "Semestres",
        "campus": "Campus",
        "abomey": "Localisation",
        "situé": "Localisation",
        "bourse": "Bourses",
        "internat": "Internat",
        "administrateur": "Administration",
        "admin": "Administration",
        "directeur": "Direction",
        "responsable": "Administration",
        "compos": "Concours",
        "composition": "Concours",
        "épreuve": "Concours"
    }
    for mot, titre in mots_cles.items():
        if mot in question_lower:
            return f"INSPEI - {titre}"
    mots = question.split()[:5]
    if len(mots) >= 2:
        return f"INSPEI - {' '.join(mots).capitalize()}"
    return "INSPEI - Nouvelle conversation"

# ---------- 8. BARRE LATÉRALE ----------
with st.sidebar:
    st.markdown("### 💬 Conversations")
    
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        nouvelle_conversation()
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.conversations:
        for convo in st.session_state.conversations:
            is_active = convo["id"] == st.session_state.current_convo_id
            display_title = convo["titre"]
            if len(display_title) > 25:
                display_title = display_title[:25] + "..."
            if st.button(
                f"{'📌' if is_active else '💬'} {display_title}",
                key=f"convo_{convo['id']}",
                use_container_width=True
            ):
                charger_conversation(convo["id"])
                st.rerun()
    else:
        st.info("Aucune conversation active.")

# ---------- 9. ZONE PRINCIPALE ----------
if not st.session_state.conversations:
    nouvelle_conversation("INSPEI - Bienvenue")

active_convo = get_active_conversation()
if active_convo is None:
    st.warning("⚠️ Aucune conversation sélectionnée.")
    st.stop()

for message in active_convo["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- 10. INPUT ----------
if prompt := st.chat_input("Posez votre question..."):
    active_convo["messages"].append({"role": "user", "content": prompt})
    if len(active_convo["messages"]) == 1:
        active_convo["titre"] = generer_titre(prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            historique = active_convo["messages"][:-1] if active_convo["messages"] else []
            reponse = repondre(prompt, model, index, data, historique)
            st.markdown(reponse)
    
    active_convo["messages"].append({"role": "assistant", "content": reponse})
    st.rerun()

# ---------- 11. PIED DE PAGE ----------
st.markdown("""
<div class="footer">
    INSPEI &bull; Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur
</div>
""", unsafe_allow_html=True)
