import os
from dotenv import load_dotenv

# Je charge le fichier .env
load_dotenv()

# Je récupère la clé API
ma_cle = os.getenv("GROQ_API_KEY")

# Je vérifie si ça a marché
if ma_cle:
    print("✅ Clé API chargée avec succès !")
else:
    print("❌ ERREUR : La clé API n'est pas dans le .env")