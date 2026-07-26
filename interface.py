import streamlit as st
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq
from spellchecker import SpellChecker

# ---------- 1. CONFIGURATION ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- 2. CORRECTEUR ORTHOGRAPHIQUE ----------
spell = SpellChecker(language='fr')

def corriger_orthographe(texte):
    if not texte or len(texte.strip()) <= 2:
        return texte
    mots = texte.split()
    mots_corriges = []
    for mot in mots:
        mot_propre = mot.strip('.,;!?()[]{}"\'')
        if mot_propre and len(mot_propre) > 1 and not mot_propre.isnumeric():
            correction = spell.correction(mot_propre)
            if correction and correction != mot_propre:
                if mot != mot_propre:
                    ponctuation = mot.replace(mot_propre, '')
                    mots_corriges.append(correction + ponctuation)
                else:
                    mots_corriges.append(correction)
            else:
                mots_corriges.append(mot)
        else:
            mots_corriges.append(mot)
    return ' '.join(mots_corriges)

# ---------- 3. CHARGEMENT DU SYSTÈME ----------
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

# ---------- 4. RECHERCHE ----------
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

# ---------- 5. SALUTATIONS ----------
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

# ---------- 6. RÉPONSE PRINCIPALE ----------
def repondre(question, model, index, data, historique=[]):
    # 0. Salutations
    if est_salutation(question) and len(historique) == 0:
        return reponse_salutation()
    
    # --- CORRECTION ORTHOGRAPHIQUE ---
    question_corrigee = corriger_orthographe(question)
    if question_corrigee != question:
        question = question_corrigee
    
    question_lower = question.lower().strip()
    
    # ============================================================
    # 1. RÈGLES SPÉCIFIQUES (prioritaires)
    # ============================================================
    
    # --- RÈGLE : confirmations ---
    mots_confirmation = ["ok", "oui", "non", "merci", "d'accord", "super", "parfait", "cool", "okay", "yes", "no", "si", "sisi"]
    if question_lower.strip() in mots_confirmation:
        return "Parfait ! 😊 N'hésitez pas si vous avez d'autres questions sur l'INSPEI, les admissions, les filières ou les écoles d'ingénieurs."
    
    # --- RÈGLE : "c'est tout" ---
    if "c'est tout" in question_lower or "cest tout" in question_lower:
        return "Oui, c'est tout pour ce sujet. 😊 Si vous voulez plus d'informations sur un point précis (admission, concours, matières, vie étudiante...), n'hésitez pas à me demander !"
    
    # --- RÈGLE : "repete" ---
    if question_lower in ["repete", "répète", "repetes", "répètes", "repeter", "répéter"]:
        return "Je suis à votre disposition pour toute question sur l'INSPEI. Que souhaitez-vous savoir ?"
    
    # --- RÈGLE : "Inspei" seul ---
    if question_lower.strip() in ["inspei", "inspéi", "insp"]:
        return "L'INSPEI est l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur. C'est une école préparatoire aux grandes écoles d'ingénieurs du Bénin, située à Abomey. Que souhaitez-vous savoir ? (Admission, filières, écoles, concours, vie étudiante...) 😊"
    
    # --- RÈGLE : "C'est quoi" ---
    if ("quoi" in question_lower or "definition" in question_lower or "c'est quoi" in question_lower or "qu'est-ce" in question_lower) and "inspei" in question_lower:
        return "L'INSPEI est l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur. C'est un établissement public rattaché à l'UNSTIM (Université Nationale des Sciences, Technologies, Ingénierie et Mathématiques). Il a été officiellement créé par l'arrêté N°719/MESRS/... du 23/12/2020, mais a démarré ses activités dès 2016-2017. Sa mission est de former des bacheliers scientifiques pour les grandes écoles d'ingénieurs du Bénin. La formation dure deux ans et débouche sur le CPEI."
    
    # --- RÈGLE : "Comment aller et c'est où" (itinéraire + localisation) - PRIORITAIRE ---
    if ("comment aller" in question_lower or "se rendre" in question_lower) and ("ou" in question_lower or "où" in question_lower):
        return "📍 **Comment se rendre à l'INSPEI et où se trouve-t-il ?**\n\n**Où se trouve l'INSPEI ?**\nL'INSPEI est situé en République du Bénin, dans le Département du Zou, à **Abomey**, à environ 1 km de la place Goho, sur la route RNIE2 en allant vers Bohicon, à Sogbo-Aliho.\n\n**Comment y aller ?**\n\n🚗 **En voiture / taxi** : Prenez la route RNIE2 vers Bohicon. L'INSPEI est à gauche, à environ 1 km de la place Goho.\n\n🛵 **En taxi-moto (zémidjan)** : Dites au conducteur 'INSPEI, quartier Sogbo-Aliho' (c'est bien connu).\n\n🚌 **En bus / taxi-brousse** : Descendez à Abomey, puis prenez un taxi-moto jusqu'à l'INSPEI.\n\n📍 **Repère** : L'école est située dans le quartier Sogbo-Aliho, près de l'ENEAM d'Abomey.\n\n💡 **Depuis Cotonou** : Prenez un bus ou taxi-brousse direction Abomey, puis suivez les indications ci-dessus."
    
    # --- RÈGLE : "Devenir étudiant" ---
    if ("etudiant" in question_lower or "étudiant" in question_lower or "inscrire" in question_lower or "inscription" in question_lower or "postuler" in question_lower) and ("inspei" in question_lower or "là bas" in question_lower or "la bas" in question_lower):
        return "📝 **Comment devenir étudiant à l'INSPEI :**\n\nL'entrée à l'INSPEI se fait **exclusivement sur concours**. Voici les étapes :\n\n📌 **Étape 1 - Vérifier les conditions** :\n• Avoir 12/20 minimum au baccalauréat\n• Être âgé de moins de 22 ans au 31 décembre 2026\n\n📌 **Étape 2 - S'inscrire en ligne** :\n• Site : www.concours.enseignementsuperieur.gouv.bj\n• Période : début août 2026\n• Frais : 5000 FCFA\n\n📌 **Étape 3 - Déposer le dossier** :\n• Centres : INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, ENS Porto-Novo\n\n📌 **Étape 4 - Passer le concours** :\n• Date : jeudi 10 septembre 2026\n• Matières : Mathématiques, Physique, Chimie, Technologie\n\n📌 **Étape 5 - Résultats et sélection**\n\n💡 L'inscription seule ne suffit pas."
    
    # --- RÈGLE : "On devient quoi après" ---
    if ("on devient quoi" in question_lower or 
        "que faire après" in question_lower or 
        "après on fait quoi" in question_lower or
        "devenir quoi" in question_lower or
        ("après" in question_lower and "devenir" in question_lower) or
        ("débouché" in question_lower) or
        ("que faire" in question_lower and "après" in question_lower)):
        return "🎓 **Que faire après l'INSPEI ?**\n\nAprès vos deux années de classes préparatoires à l'INSPEI, vous obtenez le **CPEI** (Certificat Préparatoire aux Études d'Ingénieur).\n\n📌 **Vous pouvez intégrer les écoles d'ingénieurs de l'UNSTIM :**\n\n🏛️ **ENSGEP** (Génie Energétique et Procédés)\n• Spécialités : Énergie, Thermique, Procédés industriels\n• Débouchés : Ingénieur en énergie, Chef d'usine, Bureau d'études\n\n🏛️ **ENSGMM** (Génie Mathématique et Modélisation)\n• Spécialités : Modélisation, Statistique-Finance, Informatique-Logistique\n• Débouchés : Data Scientist, Analyste financier, Logistique\n\n🏛️ **ENSTP** (Travaux Publics)\n• Spécialités : Génie civil, Construction, Infrastructures\n• Débouchés : Ingénieur BTP, Bureau d'études, Collectivités\n\n📌 **Cycle d'ingénieur** : 3 ans après l'INSPEI pour obtenir le diplôme d'ingénieur (Bac+5).\n\n💡 **Les débouchés sont nombreux** : Industrie, Énergie, Finance, Construction, Recherche...\n\n📍 **Adresse** : Abomey, quartier Sogbo-Aliho\n🌐 **Plus d'infos** : https://siteinspei.netlify.app"
    
    # --- RÈGLE : "Comment se préparer / avoir les épreuves" ---
    if ("préparer" in question_lower or "preparer" in question_lower or "réviser" in question_lower or "reviser" in question_lower or "annales" in question_lower or "s'entraîner" in question_lower or "entraîner" in question_lower) and ("concours" in question_lower or "inspei" in question_lower):
        return "📚 **Comment se préparer au concours INSPEI ?**\n\nVoici les ressources disponibles pour vous préparer :\n\n📌 **Annales des concours** (2017 à 2024) :\n• Mathématiques\n• Physique-Chimie-Technologie\n\n🌐 **Où les trouver ?**\n• Sur le site officiel des concours : www.concours.enseignementsuperieur.gouv.bj\n• Sur le site officiel de l'INSPEI : https://siteinspei.netlify.app (rubrique Ressources)\n\n📝 **Conseils de préparation :**\n• Réviser régulièrement les matières : Mathématiques, Physique, Chimie, Technologie\n• S'entraîner avec les annales des années précédentes\n• Travailler la gestion du temps (épreuves chronométrées)\n• Participer à des groupes de révision entre candidats\n\n📅 **Date du concours** : jeudi 10 septembre 2026\n\n💡 **Les annales sont disponibles gratuitement en ligne.**"
    
    # --- RÈGLE : "Épreuves du concours" (VERSION SIMPLIFIÉE) ---
    if "epreuve" in question_lower or "épreuve" in question_lower:
        if "concours" in question_lower or "composer" in question_lower or "compos" in question_lower or "compose" in question_lower or "composition" in question_lower:
            return "📚 **Épreuves du concours INSPEI 2026 :**\n\nVous composez **2 épreuves écrites** :\n\n📌 **Épreuve 1** : Mathématiques\n📌 **Épreuve 2** : Sciences Physiques, Chimie et Technologie\n\n📅 Date : jeudi 10 septembre 2026\n🌐 Inscription : www.concours.enseignementsuperieur.gouv.bj\n📖 Plus d'infos : https://siteinspei.netlify.app"
    
    # --- RÈGLE : "Matières enseignées à l'INSPEI" (FORMATION) ---
    if "matiere" in question_lower or "matière" in question_lower:
        if "semestre" in question_lower or "programme" in question_lower or "formation" in question_lower:
            return "📚 **Matières enseignées à l'INSPEI :**\n\nLa formation dure **2 ans** (4 semestres).\n\n📌 **Semestre 1 (S1)** :\n• Algorithmique (Algo)\n• Thermodynamique\n• Mathématiques 1\n• Chimie de l'Ingénieur\n• EPS (Education Physique et Sportive)\n• TEMC (Techniques d'Expression et de Communication)\n• Probabilités et Statistiques\n• Statique Graphique et Analytique\n\n📌 **Semestre 2 (S2)** :\n• Analyse Numérique\n• Graphe et Optimisation\n• Mathématiques 2\n• Cinématique et Dynamique\n• Langage (C, Python)\n• RDM (Résistance des Matériaux)\n• Normes et Mesures\n• Anglais technique\n\n📌 **Semestre 3 (S3)** :\n• TEMC (Techniques d'Expression et de Communication)\n• Recherche Opérationnelle\n• Mécanique des Fluides\n• Mathématiques 3\n• Physique des Matériaux\n• Géométrie Descriptive\n• Dessin Technique et DAO\n• Électricité Générale\n\n📌 **Semestre 4 (S4)** :\n• Mathématiques 4\n• Matlab\n• MPA (Modélisation des Phénomènes Aléatoires)\n• Sciences Biologiques pour l'Ingénieur\n• Transfert Thermique\n• Ondes Électromagnétiques\n• Anglais Technique Avancé\n• EPS (Education Physique et Sportive)\n\n📖 Plus d'infos : https://siteinspei.netlify.app"
    
    # --- RÈGLE : "Comment aller à INSPEI" (itinéraire) ---
    if ("comment aller" in question_lower or "comment se rendre" in question_lower or "comment venir" in question_lower) and "inspei" in question_lower:
        return "📍 **Comment se rendre à l'INSPEI :**\n\nL'INSPEI est situé à **Abomey, quartier Sogbo-Aliho**, à environ **1 km de la place Goho** sur la **route RNIE2** en direction de Bohicon.\n\n🚗 **En voiture / taxi** : Prenez la route RNIE2 vers Bohicon. L'INSPEI est à gauche, à environ 1 km de la place Goho.\n\n🛵 **En taxi-moto (zémidjan)** : Dites 'INSPEI, quartier Sogbo-Aliho' (c'est bien connu).\n\n🚌 **En bus / taxi-brousse** : Descendez à Abomey, puis prenez un taxi-moto jusqu'à l'INSPEI.\n\n📍 **Repère** : L'école est située dans le quartier Sogbo-Aliho, près de l'ENEAM d'Abomey.\n\n💡 Si vous venez de Cotonou, prenez un bus ou taxi-brousse direction Abomey, puis suivez les indications ci-dessus."
    
    # --- RÈGLE : "Où" (localisation) ---
    if ("ou" in question_lower or "où" in question_lower or "situé" in question_lower or "adresse" in question_lower) and "inspei" in question_lower:
        if "concours" in question_lower or "epreuve" in question_lower or "compos" in question_lower:
            return "📅 **Lieu du concours INSPEI 2026** :\n\n📍 Abomey : ENSTP/UNSTIM\n📍 Cotonou : CEG Gbégamey, Collège Catholique ND des Apôtres, CEG Ste Rita, CEG les Pylônes\n📍 Parakou : IFSIO"
        else:
            return "📍 **L'INSPEI est situé** :\n\nEn République du Bénin, dans le Département du Zou, à Abomey, à environ 1 km de la place Goho, sur la route RNIE2 en allant vers Bohicon, à Sogbo-Aliho."
    
    # --- RÈGLE : "Comment on va là bas" (itinéraire) ---
    if ("comment" in question_lower or "va" in question_lower or "aller" in question_lower or "se rendre" in question_lower) and ("là bas" in question_lower or "la bas" in question_lower):
        return "📍 **Comment se rendre à l'INSPEI :**\n\nL'INSPEI est situé à **Abomey, quartier Sogbo-Aliho**, à environ **1 km de la place Goho** sur la **route RNIE2** en direction de Bohicon.\n\n🚗 En voiture : Prenez la route RNIE2 vers Bohicon. L'INSPEI est à gauche.\n🛵 En taxi-moto : Dites 'INSPEI, quartier Sogbo-Aliho'\n🚌 En bus : Descendez à Abomey, puis prenez un taxi-moto."
    
    # --- RÈGLE : "C'est quand le concours" ---
    if ("quand" in question_lower or "date" in question_lower) and "concours" in question_lower:
        return "📅 **Concours INSPEI 2026** :\n\nLa date du concours d'entrée est le **jeudi 10 septembre 2026**.\n\n📌 Conditions : 12/20 au baccalauréat et moins de 22 ans au 31/12/2026\n📌 Inscription : www.concours.enseignementsuperieur.gouv.bj"
    
    # --- RÈGLE : "Où s'inscrire" ---
    if ("s'inscrire" in question_lower or "inscription" in question_lower) and ("concours" in question_lower or "inspei" in question_lower):
        return "📝 **Inscription au concours INSPEI 2026 :**\n\n🌐 Site : www.concours.enseignementsuperieur.gouv.bj\n📌 Dépôt des dossiers : début août 2026\n📍 Centres : INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, ENS Porto-Novo"
    
    # --- RÈGLE : "Les écoles" ---
    if ("école" in question_lower or "ecole" in question_lower) and "inspei" in question_lower:
        return "🎓 **Écoles d'ingénieurs de l'UNSTIM :**\n\n🏛️ **ENSGEP** : Génie Energétique et Procédés\n🏛️ **ENSGMM** : Génie Mathématique et Modélisation\n🏛️ **ENSTP** : Travaux Publics\n\nToutes sont situées à Abomey et accessibles après l'INSPEI."
    
    # --- RÈGLE : "Les administrateurs" ---
    if "administrateur" in question_lower or "admin" in question_lower or "direction" in question_lower or "responsable" in question_lower:
        return "👨‍🏫 **Équipe dirigeante de l'INSPEI :**\n\n• Dr (MC) AKOWANOU Christian D. (Directeur)\n• Dr. Bernard N. TOKPOHOZIN (CSSE)\n• GBEGNITO Wilfried Hodonou (Secrétaire général)\n• Comptable • Chef matériel • AKPAVOU Chédrac (Conducteur de Bus)"
    
    # --- Si la question est trop courte ---
    if len(question.strip().split()) <= 1:
        return "Pouvez-vous préciser votre question sur l'INSPEI ? Je suis là pour vous renseigner sur les admissions, les filières, les écoles, la vie étudiante, etc."
    
    # ============================================================
    # 2. RECHERCHE DANS LA BASE DE DONNÉES (JSON)
    # ============================================================
    resultats = rechercher(question, model, index, data, k=3)
    
    seuil = 0.60
    if len(question.split()) <= 3:
        seuil = 0.45
    
    if resultats and resultats[0]['similarite'] > seuil:
        return resultats[0]['reponse']
    
    # ============================================================
    # 3. APPEL À GROQ (si pas de réponse dans la base)
    # ============================================================
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
    
    messages = [
        {"role": "system", "content": f"""Tu es un conseiller pédagogique expert de l'INSPEI au Bénin.

{contexte}

RÈGLES :
1. Réponds uniquement avec les informations du contexte.
2. Si l'information n'est pas dans le contexte, dis : "Je ne dispose pas de cette information. Consultez le site officiel."
3. Continue naturellement la conversation."""},
        {"role": "user", "content": question}
    ]
    
    try:
        reponse_ia = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )
        reponse_texte = reponse_ia.choices[0].message.content
        
        # --- VÉRIFICATION POST-GÉNÉRATION ---
        for item in data:
            if item['answer'].strip() in reponse_texte or reponse_texte.strip() in item['answer']:
                return item['answer']
        
        return reponse_texte
        
    except Exception as e:
        return "Désolé, une erreur s'est produite. Veuillez réessayer."

# ---------- 7. INTERFACE STREAMLIT ----------
st.set_page_config(page_title="Assistant INSPEI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .main-header h1 { font-size: 2.5rem; color: #1a3a5c; margin-bottom: 0; }
    .main-header p { font-size: 1.1rem; color: #555; margin-top: 0; }
    .footer { text-align: center; padding: 1rem 0; font-size: 0.85rem; color: #888; border-top: 1px solid #eee; margin-top: 2rem; }
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

# ---------- 8. GESTION DES CONVERSATIONS ----------
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

# ---------- 9. BARRE LATÉRALE ----------
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

# ---------- 10. ZONE PRINCIPALE ----------
if not st.session_state.conversations:
    nouvelle_conversation("INSPEI - Bienvenue")

active_convo = get_active_conversation()
if active_convo is None:
    st.warning("⚠️ Aucune conversation sélectionnée.")
    st.stop()

for message in active_convo["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- 11. INPUT ----------
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

# ---------- 12. PIED DE PAGE ----------
st.markdown("""
<div class="footer">
    INSPEI &bull; Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur
</div>
""", unsafe_allow_html=True)
