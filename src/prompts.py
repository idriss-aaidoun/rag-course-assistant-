# src/prompts.py
from langchain.prompts import PromptTemplate

# ─── Template de base ───────────────────────────────────────
# {context} = les chunks récupérés par FAISS
# {question} = la question de l'utilisateur
# Ces deux variables sont remplies automatiquement par RetrievalQA

EXPLIQUER_TEMPLATE = """Tu es un assistant pédagogique. 
En utilisant UNIQUEMENT le contexte ci-dessous, explique clairement le concept demandé.
Si l'information n'est pas dans le contexte, dis-le honnêtement.

Contexte du cours :
{context}

Question : {question}

Explication claire et structurée :"""


QCM_TEMPLATE = """Tu es un professeur qui crée des QCM.
En utilisant UNIQUEMENT le contexte ci-dessous, génère 5 questions à choix multiples (QCM).
Pour chaque question : 4 options (A, B, C, D) avec UNE seule bonne réponse.
Indique la bonne réponse à la fin de chaque question.

Contexte du cours :
{context}

Sujet du QCM : {question}

QCM :"""


FICHE_TEMPLATE = """Tu es un assistant de révision.
En utilisant UNIQUEMENT le contexte ci-dessous, crée une fiche de révision synthétique.
Structure : titre, points clés (bullet points), définitions importantes, à retenir.

Contexte du cours :
{context}

Sujet de la fiche : {question}

Fiche de révision :"""


# ─── Sélecteur de prompt ────────────────────────────────────
PROMPTS = {
    "expliquer": EXPLIQUER_TEMPLATE,
    "qcm":       QCM_TEMPLATE,
    "fiche":     FICHE_TEMPLATE,
}

def get_prompt(mode: str) -> PromptTemplate:
    """
    Retourne le bon PromptTemplate selon le mode choisi.
    Les variables input_variables doivent correspondre exactement
    aux {placeholders} dans le template.
    """
    template = PROMPTS.get(mode, EXPLIQUER_TEMPLATE)
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )