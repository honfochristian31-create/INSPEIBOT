import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ---------- ÉTAPE 1 : Charger les données du JSON ----------
def charger_donnees():
    """Charge le fichier inspei.json et prépare les textes"""
    with open('data/inspei.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # On crée une liste de textes combinant question + réponse
    textes = []
    for item in data:
        textes.append(f"Q: {item['question']} R: {item['answer']}")
    
    return data, textes

# ---------- ÉTAPE 2 : Créer l'index vectoriel ----------
def creer_index(textes):
    """Transforme les textes en vecteurs et crée un index FAISS"""
    print("🔄 Chargement du modèle IA pour comprendre le français...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🔄 Transformation des textes en chiffres (vecteurs)...")
    vecteurs = model.encode(textes, show_progress_bar=True)
    vecteurs = np.array(vecteurs).astype('float32')
    
    # Création de l'index FAISS
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    
    print(f"✅ Index créé avec {len(vecteurs)} vecteurs !")
    return model, index

# ---------- ÉTAPE 3 : Fonction de recherche ----------
def rechercher(question, model, index, data, k=3):
    """Recherche les k questions/réponses les plus proches"""
    # Transformer la question en vecteur
    vecteur_question = model.encode([question])
    vecteur_question = np.array(vecteur_question).astype('float32')
    
    # Interroger FAISS
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

# ---------- ÉTAPE 4 : Test ----------
if __name__ == "__main__":
    print("🚀 TEST DU MOTEUR DE RECHERCHE\n")
    
    # Charger les données
    data, textes = charger_donnees()
    print(f"📚 {len(data)} questions chargées depuis inspei.json\n")
    
    # Créer l'index
    model, index = creer_index(textes)
    
    # Poser une question de test
    test_question = "Où se trouve l'INSPEI ?"
    print(f"\n🔍 Recherche : '{test_question}'\n")
    
    resultats = rechercher(test_question, model, index, data, k=3)
    
    print("📊 RÉSULTATS :")
    for i, res in enumerate(resultats):
        print(f"\n--- Résultat {i+1} (similarité: {res['similarite']:.2f}) ---")
        print(f"Question : {res['question']}")
        print(f"Réponse : {res['reponse'][:150]}...")