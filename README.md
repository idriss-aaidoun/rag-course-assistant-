# 📚 Assistant de Révision de Cours

> Un assistant RAG local qui transforme vos notes et slides en outil de révision intelligent — génération de QCM, fiches de révision et explications de concepts, le tout sans connexion internet.

---

## ✨ Fonctionnalités

- 🧠 **Expliquer un concept** — reformulation claire à partir de vos cours
- 📝 **Générer des QCM** — 5 questions avec 4 options et correction automatique
- 📄 **Créer des fiches** — synthèse structurée prête à imprimer
- 🔒 **100% local** — aucune donnée envoyée à l'extérieur
- 📂 **Multi-format** — supporte PDF, PPTX, TXT, Markdown

---

## 🏗️ Architecture

```
Document PDF/Slides
        │
        ▼
  LangChain Loaders          ← Chargement des documents
        │
        ▼
  Text Splitter              ← Découpage en chunks (500 chars)
        │
        ▼
  nomic-embed-text (Ollama)  ← Vectorisation locale
        │
        ▼
  FAISS Vector Store         ← Index sauvegardé sur disque
        │
   (au moment de la question)
        │
        ▼
  Retriever (top-4 chunks)   ← Recherche sémantique
        │
        ▼
  Prompt Builder             ← QCM / Fiche / Explication
        │
        ▼
  LLM local (Ollama)         ← Génération de réponse
        │
        ▼
  Interface Gradio           ← Résultat affiché
```

---

## 🛠️ Stack technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| Interface | [Gradio](https://gradio.app) | UI web locale |
| Orchestration | [LangChain](https://langchain.com) | Pipeline RAG |
| Embedding | [nomic-embed-text](https://ollama.com) | Vectorisation locale |
| Vector Store | [FAISS](https://faiss.ai) | Recherche sémantique |
| LLM | [Ollama](https://ollama.com) (llama3) | Génération de texte |

---

## ⚙️ Installation

### Prérequis
- Python 3.10+
- [Ollama](https://ollama.com/download) installé et lancé

### 1. Cloner le repo

```bash
git clone https://github.com/idriss-aaidoun/rag-course-assistant-.git
cd rag-course-assistant-
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger les modèles Ollama

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

---

## 🚀 Utilisation

### Étape 1 — Ajouter vos cours

Déposez vos fichiers PDF dans le dossier :
```
data/cours/
```

### Étape 2 — Lancer l'ingestion (une seule fois par document)

```bash
python src/ingest.py
```

### Étape 3 — Lancer l'application

```bash
python app.py
```

Ouvrez votre navigateur sur `http://localhost:7860`

---

## 📁 Structure du projet

```
rag-course-assistant-/
├── data/
│   └── cours/              ← Vos PDF et slides ici
├── vectorstore/            ← Index FAISS généré (auto)
├── src/
│   ├── ingest.py           ← Pipeline d'ingestion RAG
│   ├── rag_chain.py        ← Chain de retrieval + LLM
│   └── prompts.py          ← Templates QCM / Fiche / Explication
├── app.py                  ← Interface Gradio
├── requirements.txt
└── README.md
```

---

## 💡 Exemple d'utilisation

```
Question  : "Explique le théorème de Bayes"
Mode      : expliquer
Réponse   : [Explication générée à partir de vos notes de cours]

Question  : "Les dérivées"
Mode      : qcm
Réponse   : [5 questions QCM avec correction]

Question  : "Résumé du chapitre 3"
Mode      : fiche
Réponse   : [Fiche de révision structurée]
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

---

## 📄 Licence

MIT License — libre d'utilisation et de modification.

---

<p align="center">
  Fait avec ❤️ pour apprendre le RAG
</p>
