# src/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Chemins ────────────────────────────────────────────────
DATA_DIR       = "data/cours"
VECTORSTORE_DIR = "vectorstore"

# ─── 1. CHARGEMENT ─────────────────────────────────────────
def load_documents():
    """
    DirectoryLoader parcourt tout le dossier data/cours/
    et charge automatiquement tous les fichiers PDF.
    Chaque page PDF devient un Document LangChain
    avec son contenu texte + ses métadonnées (nom fichier, page...).
    """
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",          # cherche tous les PDF récursivement
        loader_cls=PyPDFLoader,   # utilise PyPDF pour lire chaque fichier
        show_progress=True
    )
    documents = loader.load()
    print(f"[+] {len(documents)} pages chargées depuis {DATA_DIR}")
    return documents

# ─── 2. DÉCOUPAGE EN CHUNKS ─────────────────────────────────
def split_documents(documents):
    """
    POURQUOI découper ?
    Les LLMs ont une fenêtre de contexte limitée. On ne peut pas
    envoyer tout le cours d'un coup. On découpe en morceaux (chunks)
    pour retrouver seulement les parties PERTINENTES à la question.

    chunk_size=500   → chaque chunk fait ~500 caractères
    chunk_overlap=50 → 50 caractères partagés entre chunks voisins
                       (évite de couper une phrase en deux)

    RecursiveCharacterTextSplitter essaie de couper sur :
    \n\n (paragraphes) → \n (lignes) → . (phrases) → caractères
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[+] {len(chunks)} chunks créés")
    return chunks

# ─── 3. EMBEDDING + INDEXATION FAISS ───────────────────────
def create_vectorstore(chunks):
    """
    EMBEDDING : transformer chaque chunk de texte en vecteur numérique.
    nomic-embed-text tourne localement via Ollama (pas d'API externe).
    
    Deux chunks sémantiquement proches → vecteurs proches dans l'espace.
    C'est ça qui permet la RECHERCHE PAR SENS et pas par mots-clés.

    FAISS (Facebook AI Similarity Search) :
    - Stocke tous les vecteurs dans un index optimisé
    - Permet de retrouver les k vecteurs les plus proches
      d'un vecteur requête en millisecondes
    - save_local() → sauvegarde l'index sur disque
    """
    print("[+] Chargement du modèle d'embedding (Ollama)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("[+] Création de l'index FAISS (peut prendre quelques minutes)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[+] Index sauvegardé dans {VECTORSTORE_DIR}/")
    return vectorstore

# ─── POINT D'ENTRÉE ─────────────────────────────────────────
if __name__ == "__main__":
    docs   = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    print("\n✓ Pipeline d'ingestion terminé avec succès !")