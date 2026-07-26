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
    
    # --- RÈGLE : "Inspei" seul ---
    if question_lower.strip() in ["inspei", "inspéi", "insp"]:
        return "L'INSPEI est l'Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur. C'est une école préparatoire aux grandes écoles d'ingénieurs du Bénin, située à Abomey. Que souhaitez-vous savoir ? (Admission, filières, écoles, concours, vie étudiante...) 😊"
    
    # --- RÈGLE : "C'est quand le concours" (date uniquement) ---
    if ("quand" in question_lower or "date" in question_lower) and "concours" in question_lower:
        return "📅 **Concours INSPEI 2026** :\n\nLa date du concours d'entrée est le **jeudi 10 septembre 2026**.\n\n📌 **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31/12/2026\n📌 **Inscription** : www.concours.enseignementsuperieur.gouv.bj\n📌 **Lieux** : Abomey (ENSTP/UNSTIM), Cotonou (CEG Gbégamey, CEG Ste Rita, CEG les Pylônes), Parakou (IFSIO)"
    
    # --- RÈGLE : "Procédures à suivre" (démarches) ---
    if "procédure" in question_lower or "démarche" in question_lower or "étapes" in question_lower:
        if "inspei" in question_lower or "concours" in question_lower:
            return "📝 **Procédures à suivre pour intégrer l'INSPEI** :\n\n📌 **Étape 1 - Conditions** :\n• Avoir 12/20 minimum au baccalauréat\n• Être âgé de moins de 22 ans au 31 décembre 2026\n\n📌 **Étape 2 - Inscription** :\n• Se rendre sur www.concours.enseignementsuperieur.gouv.bj\n• Début août 2026\n\n📌 **Étape 3 - Dépôt des dossiers** :\n• Dans les centres : INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, ENS Porto-Novo\n\n📌 **Étape 4 - Concours** :\n• Épreuves écrites en Mathématiques, Physique, Chimie et Technologie\n• Date : jeudi 10 septembre 2026\n\n📌 **Étape 5 - Sélection** :\n• Basée sur les résultats du concours"
    
    # --- RÈGLE : "Inspei quand ?" ---
    if "inspei" in question_lower and "quand" in question_lower:
        return "📅 **Concours INSPEI 2026** :\n\nLe concours d'entrée à l'INSPEI aura lieu le **jeudi 10 septembre 2026**.\n\n📌 **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31 décembre 2026.\n📌 **Dépôt des dossiers** : début août 2026.\n📌 **Centres d'examen** : Abomey (ENSTP/UNSTIM), Cotonou (CEG Gbégamey, Collège Catholique ND des Apôtres, CEG Ste Rita, CEG les Pylônes) et Parakou (IFSIO).\n\nSource : Arrêté N°2026-0224/..."
    
    # --- RÈGLE : "Inscription ou concours" ---
    if ("inscrit" in question_lower or "inscription" in question_lower) and ("concours" in question_lower or "compose" in question_lower):
        return "📝 **Inscription et concours INSPEI :**\n\nL'entrée à l'INSPEI se fait **sur concours** (et pas seulement sur inscription).\n\n📌 **Étapes** :\n1. **Inscription** : se faire en ligne sur **www.concours.enseignementsuperieur.gouv.bj** (début août 2026)\n2. **Concours écrit** : épreuves en Mathématiques, Physique, Chimie et Technologie\n3. **Sélection** : basée sur les résultats du concours\n\n📌 **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31/12/2026\n\n**L'inscription seule ne suffit pas : il faut passer et réussir le concours.**"
    
    # --- RÈGLE : "Où s'inscrire / site d'inscription" ---
    if ("s'inscrire" in question_lower or "inscription" in question_lower or "site" in question_lower) and ("concours" in question_lower or "inspei" in question_lower):
        return "📝 **Inscription au concours INSPEI 2026** :\n\n🌐 **Site officiel d'inscription** : www.concours.enseignementsuperieur.gouv.bj\n\n📌 **Dépôt des dossiers** : début août 2026\n📍 **Centres de dépôt** : INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, ENS Porto-Novo\n\n📞 **Contact** : inspei@unstim.edu.bj\n🌐 **Site officiel de l'INSPEI** : https://siteinspei.netlify.app"
    
    # --- RÈGLE : "Comment aller / se rendre" (itinéraire) ---
    if ("aller" in question_lower or "se rendre" in question_lower or "venir" in question_lower or "transport" in question_lower) and ("inspei" in question_lower or "école" in question_lower):
        return "📍 **Comment se rendre à l'INSPEI** :\n\nL'INSPEI est situé à **Abomey, quartier Sogbo-Aliho**, à environ 1 km de la place Goho sur la route RNIE2 en direction de Bohicon.\n\n🚗 **En voiture / taxi** : Depuis Abomey, prenez la route RNIE2 vers Bohicon. L'INSPEI est situé à gauche, à environ 1 km de la place Goho.\n\n🛵 **En taxi-moto (zémidjan)** : Dites au conducteur de vous déposer à l'INSPEI, quartier Sogbo-Aliho (c'est bien connu).\n\n🚌 **En bus / taxi-brousse** : Descendez à Abomey, puis prenez un taxi-moto jusqu'à l'INSPEI.\n\n📌 **Repère** : L'école est dans l'enceinte de l'ENEAM d'Abomey.\n\nSi vous voulez d'autres indications, précisez votre point de départ ! 😊"
    
    # --- RÈGLE : "Épreuves et matières du concours" ---
    if ("epreuves" in question_lower or "épreuves" in question_lower or "matieres" in question_lower or "matières" in question_lower) and ("concours" in question_lower or "inspei" in question_lower):
        return "📚 **Épreuves et matières du concours INSPEI 2026** :\n\nLes épreuves du concours d'entrée à l'INSPEI portent sur les matières suivantes :\n\n📐 **Mathématiques** : Algèbre, Analyse, Géométrie, Probabilités\n⚛️ **Physique** : Mécanique, Électricité, Optique, Thermodynamique\n🧪 **Chimie** : Chimie générale, Chimie organique, Chimie des solutions\n🛠️ **Technologie** : Sciences de l'ingénieur, Mécanique, Électrotechnique\n\n📌 **Format** : Épreuves écrites\n📌 **Date** : jeudi 10 septembre 2026\n📌 **Inscription** : www.concours.enseignementsuperieur.gouv.bj"
    
    # --- RÈGLE : "C'est où" (sans mention d'INSPEI) ---
    if question_lower in ["c'est où", "cest ou", "c est ou", "c'est ou"]:
        return "Si vous cherchez la localisation de l'INSPEI, il est situé à Abomey, quartier Sogbo-Aliho, à environ 1 km de la place Goho sur la route RNIE2. Si vous cherchez autre chose, précisez votre question."
    
    # --- RÈGLE : SUIVI CONTEXTUEL ---
    if len(question.strip().split()) <= 2 and historique:
        dernier_echange = historique[-1] if historique else None
        dernier_sujet = dernier_echange["content"].lower() if dernier_echange else ""
        
        if question_lower in ["ou", "où", "ou ça", "où ça", "ou se passe", "où se passe"]:
            if "concours" in dernier_sujet or "composition" in dernier_sujet or "epreuve" in dernier_sujet:
                return "📅 **Lieu du concours INSPEI 2026** :\n\nLes épreuves se déroulent dans les centres suivants :\n📍 **Abomey** : ENSTP/UNSTIM\n📍 **Cotonou** : CEG Gbégamey, Collège Catholique ND des Apôtres, CEG Ste Rita, CEG les Pylônes\n📍 **Parakou** : IFSIO\n\n📌 **Date** : jeudi 10 septembre 2026"
            elif "inspei" in dernier_sujet or "école" in dernier_sujet:
                return "L'INSPEI est situé à Abomey, quartier Sogbo-Aliho, à environ 1 km de la place Goho sur la route RNIE2."
            else:
                return "Pouvez-vous préciser de quoi vous parlez ? (concours, école, événement, etc.)"
        
        if question_lower in ["là bas", "y aller", "comment y aller", "comment s'y rendre"]:
            if "inscription" in dernier_sujet or "concours" in dernier_sujet or "déposer" in dernier_sujet:
                return "📝 **Procédures d'inscription à l'INSPEI** :\n\n📌 **Étape 1** : Vérifier les conditions (12/20 au bac, moins de 22 ans)\n📌 **Étape 2** : S'inscrire en ligne sur www.concours.enseignementsuperieur.gouv.bj (début août 2026)\n📌 **Étape 3** : Déposer le dossier dans un centre (INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, ENS Porto-Novo)\n📌 **Étape 4** : Passer les épreuves écrites (jeudi 10 septembre 2026)\n📌 **Étape 5** : Attendre les résultats\n\n💡 **L'adresse de l'INSPEI** : Abomey, quartier Sogbo-Aliho, à 1 km de la place Goho."
            elif "inspei" in dernier_sujet or "école" in dernier_sujet:
                return "📍 **Comment se rendre à l'INSPEI** :\n\nL'INSPEI est situé à Abomey, quartier Sogbo-Aliho, à environ 1 km de la place Goho sur la route RNIE2 (direction Bohicon).\n\n🚗 **En voiture** : Suivez la RNIE2 depuis Abomey vers Bohicon.\n🛵 **En taxi-moto** : Demandez à être déposé à l'INSPEI, quartier Sogbo-Aliho.\n📌 **Repère** : L'école est dans l'enceinte de l'ENEAM d'Abomey."
            else:
                return "📝 **Procédures pour intégrer l'INSPEI** :\n\n📌 **Étape 1 - Conditions** : 12/20 au bac et moins de 22 ans\n📌 **Étape 2 - Inscription** : www.concours.enseignementsuperieur.gouv.bj\n📌 **Étape 3 - Dépôt des dossiers** : début août 2026\n📌 **Étape 4 - Concours** : jeudi 10 septembre 2026\n\n📍 **L'adresse** : Abomey, quartier Sogbo-Aliho"
        
        if question_lower in ["quand", "et quand", "à quelle date", "date"]:
            if "concours" in dernier_sujet or "composition" in dernier_sujet or "epreuve" in dernier_sujet:
                return "Le concours INSPEI 2026 aura lieu le **jeudi 10 septembre 2026**."
            elif "dossiers" in dernier_sujet or "inscription" in dernier_sujet or "déposer" in dernier_sujet or "dépôt" in dernier_sujet:
                return "📅 **Dépôt des dossiers pour le concours INSPEI 2026** :\n\nLe dépôt des dossiers sera effectif en **début août 2026**.\n\n📍 **Centres de dépôt** : INSPEI Abomey, ENS Natitingou, IFSIO Parakou, ENSET Lokossa, INMeS Cotonou, et ENS Porto-Novo.\n\n📌 **Inscription en ligne** : www.concours.enseignementsuperieur.gouv.bj"
            elif "inspei" in dernier_sujet or "école" in dernier_sujet:
                return "La formation à l'INSPEI dure deux ans, organisée en quatre semestres."
            else:
                return "Pouvez-vous préciser de quoi vous parlez ? (concours, inscription, formation...)"

        if question_lower in ["et", "et quoi", "quoi d'autre", "autre chose"]:
            if "concours" in dernier_sujet:
                return "📌 **Plus d'informations sur le concours INSPEI 2026** :\n\n• **Dépôt des dossiers** : début août 2026\n• **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31/12/2026\n• **Inscription** : www.concours.enseignementsuperieur.gouv.bj\n• **Frais** : 5000 FCFA"
            elif "inspei" in dernier_sujet or "école" in dernier_sujet:
                return "📌 **Plus d'informations sur l'INSPEI** :\n\n• **Filière** : Classes Préparatoires (CPEI)\n• **Durée** : 2 ans\n• **Débouchés** : ENSGEP, ENSGMM, ENSTP\n• **Hébergement** : Internat avec bourse"
            else:
                return "Que voulez-vous savoir d'autre ?"
    
    # --- RÈGLE : "travailler" (conseils pour les études) ---
    if "travailler" in question_lower or ("étude" in question_lower and "réussir" in question_lower):
        return "Pour réussir à l'INSPEI, voici quelques conseils :\n\n📚 **Organisez-vous** : établissez un emploi du temps quotidien.\n📝 **Révisez régulièrement** : les classes préparatoires exigent un travail constant.\n👨‍🏫 **Demandez de l'aide** : n'hésitez pas à solliciter vos enseignants.\n⏰ **Prenez des pauses** : le repos est essentiel.\n🎯 **Fixez-vous des objectifs** : restez motivé pour les concours.\n\nL'INSPEI est exigeant, mais avec de la discipline, vous réussirez ! 💪"
    
    # --- RÈGLE : "transfert" ---
    if "transfert" in question_lower or "transférer" in question_lower or "changer d'école" in question_lower:
        return "Si vous souhaitez des informations sur un transfert vers l'INSPEI ou un changement d'établissement, je vous invite à contacter directement le secrétariat de l'INSPEI (inspei@unstim.edu.bj) ou à consulter le site officiel. Les modalités de transfert sont gérées au cas par cas par l'administration."
    
    # --- RÈGLE : "COMMENT FAIRE" ---
    mots_comment = ["comment faire", "comment je fais", "comment puis-je", "que dois-je", "je fais comment", "comment procéder"]
    if any(mot in question_lower for mot in mots_comment):
        if "info" in question_lower or "plus" in question_lower:
            return "Pour obtenir plus d'informations, vous pouvez :\n\n🌐 Consulter le site officiel : https://siteinspei.netlify.app\n📧 Envoyer un email à : inspei@unstim.edu.bj\n📞 Contacter l'INSPEI au : +229 97692697 / +229 67850182\n📍 Vous rendre à l'adresse : Abomey, quartier Sogbo-Aliho\n📝 Vous inscrire sur : www.concours.enseignementsuperieur.gouv.bj\n\nOu posez-moi une question précise sur les matières, l'admission, les écoles, etc. ! 😊"
    
    # --- RÈGLE : "PLUS D INFOS" ---
    mots_plus_infos = ["plus d'infos", "plus d info", "plus d'informations", "plus dinfos", "en savoir plus", "plus de details", "plus de détails"]
    if question_lower in mots_plus_infos or ("plus" in question_lower and ("info" in question_lower or "detail" in question_lower)):
        return "Voici les coordonnées et adresses utiles pour obtenir plus d'informations sur l'INSPEI :\n\n🌐 **Site officiel** : https://siteinspei.netlify.app\n📧 **Email** : inspei@unstim.edu.bj\n📍 **Adresse physique** : République du Bénin, Département du Zou, Abomey, quartier Sogbo-Aliho (à environ 1 km de la place Goho sur la route RNIE2)\n📝 **Site d'inscription aux concours** : www.concours.enseignementsuperieur.gouv.bj\n📞 **Téléphone** : +229 97692697 / +229 67850182\n\nSi vous avez besoin d'informations plus précises sur un sujet spécifique (matières, admission, écoles, etc.), n'hésitez pas à me poser la question directement ! 😊"
    
    # --- RÈGLE PRIORITAIRE : "C'EST QUOI" ---
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
    
    # --- RÈGLE POUR LES MATIÈRES DE LA FORMATION ---
    if question_lower in ["les matieres", "matieres", "les matière", "matière"] and "concours" not in question_lower:
        return "Les matières enseignées à l'INSPEI sont réparties sur 4 semestres. Voici le programme :\n\nSemestre 1 : Algorithmique, Thermodynamique, Maths 1, Chimie de l'Ingénieur, EPS, TEMC, Probabilités et Statistiques, Statique Graphique et Analytique.\n\nSemestre 2 : Analyse Numérique, Graphe et Optimisation, Maths 2, Cinématique et Dynamique, Langage (C/Python), RDM, Normes et Mesures, Anglais technique.\n\nSemestre 3 : TEMC, Recherche Opérationnelle, Mécanique des Fluides, Maths 3, Physique des Matériaux, Géométrie Descriptive, Dessin Technique et DAO, Électricité Générale.\n\nSemestre 4 : Maths 4, Matlab, MPA, Sciences Biologiques pour l'Ingénieur, Transfert Thermique, Ondes Électromagnétiques, Anglais Technique Avancé, EPS."
    
    # --- RÈGLE : "COMPOS" ---
    mots_compos = ["compos", "composition", "épreuve", "compose", "epreuve", "concours écrit", "on compose", "quand on compose", "date des épreuves", "composition"]
    if any(mot in question_lower for mot in mots_compos):
        if "quand" in question_lower or "date" in question_lower or "on compose" in question_lower or "composition" in question_lower:
            return "📅 **Date des épreuves du concours INSPEI 2026** :\n\nLes épreuves écrites se dérouleront le **jeudi 10 septembre 2026**.\n\n📍 **Centres d'examen** : Abomey (ENSTP/UNSTIM), Cotonou (CEG Gbégamey, Collège Catholique ND des Apôtres, CEG Ste Rita, CEG les Pylônes) et Parakou (IFSIO).\n\n📝 **Matières évaluées** : Mathématiques, Physique, Chimie et Technologie.\n\n📌 **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31 décembre 2026.\n\n📌 **Dépôt des dossiers** : début août 2026.\n\nPour plus de détails, consultez l'avis de concours sur www.concours.enseignementsuperieur.gouv.bj."
        else:
            return "Les épreuves du concours d'entrée à l'INSPEI se déroulent en centres d'examen : Abomey (ENSTP/UNSTIM), Cotonou (CEG Gbégamey, Collège Catholique ND des Apôtres, CEG Ste Rita, CEG les Pylônes) et Parakou (IFSIO). Les matières évaluées sont les Mathématiques, la Physique, la Chimie et la Technologie. Consultez l'avis de concours officiel pour les détails précis de l'année en cours sur www.concours.enseignementsuperieur.gouv.bj."
    
    # --- RÈGLE : "DATE CONCOURS" ---
    mots_date = ["date du concours", "concours date", "quand a lieu", "date concours", "à quelle date", "calendrier concours"]
    if any(mot in question_lower for mot in mots_date):
        return "📅 **Calendrier des concours 2026-2027** :\n\n**Jeudi 10 septembre 2026** : INMES, IFSIO, ENSPD, ENSTIC, ENEAM, IUEP-MA, INSPEI, INEPS.\n\n**Vendredi 11 septembre 2026** : ENS Porto-Novo, ENS Natitingou, ENSET Lokossa.\n\n📌 **Condition** : 12/20 au baccalauréat et moins de 22 ans au 31 décembre 2026.\n📌 **Dépôt des dossiers** : début août 2026.\n\nSource : Arrêté N°2026-0224/MESRS/..."
    
    # --- RÈGLE : "Le concours est où" ---
    if ("concours" in question_lower or "composition" in question_lower or "epreuve" in question_lower) and ("ou" in question_lower or "où" in question_lower):
        return "📅 **Lieu du concours INSPEI 2026** :\n\nLes épreuves du concours se dérouleront dans les centres d'examen suivants :\n\n📍 **Abomey** : ENSTP/UNSTIM\n📍 **Cotonou** : CEG Gbégamey, Collège Catholique Notre Dame des Apôtres, CEG Ste Rita, CEG les Pylônes\n📍 **Parakou** : IFSIO\n\n📌 **Date** : jeudi 10 septembre 2026\n📌 **Conditions** : 12/20 au baccalauréat et moins de 22 ans au 31/12/2026"
    
    # --- Si la question est trop courte (1 mot) ---
    if len(question.strip().split()) <= 1:
        return "Pouvez-vous préciser votre question sur l'INSPEI ? Je suis là pour vous renseigner sur les admissions, les filières, les écoles, la vie étudiante, etc."
    
    # --- RECHERCHE NORMALE ---
    resultats = rechercher(question, model, index, data, k=3)
    
    seuil = 0.60
    if len(question.split()) <= 3:
        seuil = 0.45
    
    if resultats and resultats[0]['similarite'] > seuil:
        return resultats[0]['reponse']
    
    # --- GROQ ---
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
