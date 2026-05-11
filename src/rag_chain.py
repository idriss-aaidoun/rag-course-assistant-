# src/rag_chain.py
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.prompts import get_prompt

# ─── Chemins ────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "mistral"          # change selon ce que tu as dans Ollama

# ─── 1. CHARGEMENT DE L'INDEX FAISS ────────────────────────
def load_vectorstore():
    """
    On recharge l'index FAISS qu'on a créé avec ingest.py.
    allow_dangerous_deserialization=True est requis par LangChain
    pour charger un fichier .pkl local (c'est notre propre fichier,
    donc pas de risque).
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("[+] Index FAISS chargé avec succès")
    return vectorstore

# ─── 2. CRÉATION DU RETRIEVER ───────────────────────────────
def get_retriever(vectorstore, k=4):
    """
    Le retriever transforme le vectorstore en "chercheur".
    Quand on lui donne une question, il :
      1. Embed la question avec nomic-embed-text
      2. Cherche les k chunks les plus proches dans FAISS
      3. Retourne ces k chunks comme contexte

    k=4 → on récupère les 4 passages les plus pertinents.
    Plus k est grand, plus de contexte mais plus lent.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

# ─── 3. CONSTRUCTION DE LA CHAIN RAG ───────────────────────
def build_chain(mode="expliquer"):
    """
    RetrievalQA assemble tout le pipeline :
    
    Question ──► Retriever ──► [chunk1, chunk2, chunk3, chunk4]
                                         │
                                         ▼
                              Prompt = instruction + contexte + question
                                         │
                                         ▼
                                   LLM (llama3)
                                         │
                                         ▼
                                      Réponse

    chain_type="stuff" → on "stuff" (fourre) tous les chunks
    dans un seul prompt. Simple et efficace pour des chunks courts.

    return_source_documents=True → la chain retourne aussi
    quels passages du cours ont été utilisés (utile pour débugger).
    """
    vectorstore = load_vectorstore()
    retriever   = get_retriever(vectorstore)
    llm         = OllamaLLM(model=LLM_MODEL, temperature=0.3)
    prompt      = get_prompt(mode)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

# ─── 4. FONCTION PRINCIPALE D'APPEL ────────────────────────
def ask(question: str, mode: str = "expliquer") -> dict:
    """
    Fonction principale appelée depuis app.py (Gradio).
    
    Retourne un dict avec :
      - "answer"  : la réponse du LLM
      - "sources" : les passages du cours utilisés
    """
    chain  = build_chain(mode)
    result = chain.invoke({"query": question})

    return {
        "answer":  result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }


# ─── TEST EN CONSOLE ────────────────────────────────────────
if __name__ == "__main__":
    question = input("Pose ta question : ")
    mode     = input("Mode (expliquer / qcm / fiche) : ")
    result   = ask(question, mode)

    print("\n── Réponse ──────────────────────────────")
    print(result["answer"])
    print("\n── Sources utilisées ────────────────────")
    for s in result["sources"]:
        print(f"  • {s}")