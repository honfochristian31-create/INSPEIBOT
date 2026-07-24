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

# ---------- LES FONCTIONS DU MOTEUR DE RECHERCHE (on les réutilise) ----------
def charger_donnees():
    with open('data/inspei.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    textes = []
    for item in data:
        textes.append(f"Q: {item['question']} R: {item['answer']}")
    return data, textes

def creer_index(textes):
    print("🔄 Chargement du modèle...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("🔄 Création des vecteurs...")
    vecteurs = model.encode(textes, show_progress_bar=True)
    vecteurs = np.array(vecteurs).astype('float32')
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    print(f"✅ Index créé avec {len(vecteurs)} vecteurs !")
    return model, index

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

# ---------- LE CHATBOT (avec Groq) ----------
def repondre(question, model, index, data):
    # 1. Chercher dans la base de connaissances
    resultats = rechercher(question, model, index, data, k=3)
    
    # 2. Si le meilleur résultat est très bon (> 0.70), on le donne directement
    if resultats and resultats[0]['similarite'] > 0.70:
        print("✅ Réponse trouvée dans la base de connaissances !")
        return resultats[0]['reponse']
    
    # 3. Sinon, on prépare un contexte avec les 3 meilleurs résultats
    contexte = ""
    for i, res in enumerate(resultats):
        contexte += f"Document {i+1}:\n"
        contexte += f"Question: {res['question']}\n"
        contexte += f"Réponse: {res['reponse']}\n\n"
    
    # 4. On demande à Groq de générer une réponse en s'inspirant du contexte
    print("🤖 Génération d'une réponse par l'IA Groq...")
    
    messages = [
        {"role": "system", "content": f"""Tu es un assistant expert de l'INSPEI (Institut National Supérieur des Classes Préparatoires aux Etudes d'Ingénieur) au Bénin.
        
Voici des informations disponibles sur l'INSPEI :
{contexte}

RÈGLES :
- Réponds uniquement en utilisant les informations du contexte.
- Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information dans ma base. Contacte le secrétariat de l'INSPEI pour plus de détails."
- Sois clair, précis, et adapté à des étudiants béninois."""},
        {"role": "user", "content": question}
    ]
    
    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return reponse.choices[0].message.content

# ---------- TEST ----------
if __name__ == "__main__":
    print("🚀 CHATBOT INSPEI - MODE TEST\n")
    
    data, textes = charger_donnees()
    model, index = creer_index(textes)
    
    print("\n💬 Pose ta question (tape 'quit' pour arrêter) :")
    
    while True:
        question = input("\n🔍 Vous : ")
        if question.lower() == 'quit':
            break
        
        reponse = repondre(question, model, index, data)
        print(f"\n🤖 Bot : {reponse}")