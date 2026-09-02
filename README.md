# 🧠 Obsidian Vault RAG Knowledge Assistant

An AI-powered assistant that lets you upload your Obsidian vault (markdown notes), indexes them into a vector database, and enables **conversational Q&A** with source citations. Ask questions about your notes and get AI-generated answers grounded in your personal knowledge base.

![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Google_Gemini-4285F4?style=flat-square&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6B35?style=flat-square)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat-square)

---

## ✨ Features

- **🔍 RAG Pipeline** — Retrieval-Augmented Generation for accurate, sourced answers
- **📝 Obsidian-Aware** — Parses `[[wikilinks]]`, `#tags`, `![[embeds]]`, and frontmatter
- **💬 Conversational Chat** — Multi-turn Q&A with conversation memory
- **📚 Vault Explorer** — Browse, search, and filter your indexed notes
- **📊 Analytics Dashboard** — Tag frequency, note connections, chunk statistics
- **🔗 Source Citations** — Every answer cites which notes it drew from
- **📦 ZIP Upload** — Upload entire Obsidian vaults as .zip files
- **⚡ Fast** — Powered by Gemini 1.5 Flash for quick responses

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Upload   │  │  Chat        │  │  Vault     │ │
│  │  Vault    │  │  Interface   │  │  Explorer  │ │
│  └─────┬────┘  └──────┬───────┘  └────────────┘ │
├────────┼───────────────┼──────────────────────────┤
│        ▼               ▼         Backend          │
│  ┌──────────┐  ┌──────────────┐                   │
│  │ Document  │  │   RAG Chain  │                   │
│  │ Processor │  │  (LangChain) │                   │
│  └─────┬────┘  └──────┬───────┘                   │
│        ▼               ▼                           │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Chunker  │  │  Retriever   │  │  Gemini API │ │
│  │ + Embed  │  │  (Top-K)     │  │  (Generate) │ │
│  └─────┬────┘  └──────┬───────┘  └─────────────┘ │
│        ▼               ▼                           │
│  ┌─────────────────────────────┐                   │
│  │       ChromaDB              │                   │
│  │   (Vector Database)         │                   │
│  └─────────────────────────────┘                   │
└──────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- A Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/obsidian-vault-rag.git
cd obsidian-vault-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the app
streamlit run app.py
```

### Demo

1. Start the app with `streamlit run app.py`
2. Enter your Gemini API key in the sidebar
3. Use the **sample vault** files in `sample_vault/` to test
4. Ask questions like:
   - "What are transformers?"
   - "Summarize my notes on neural networks"
   - "What did I learn on August 1st?"

## 📁 Project Structure

```
compiler_project/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
│
├── src/                      # Core backend
│   ├── document_processor.py # Markdown + Obsidian parsing
│   ├── chunker.py           # Text chunking strategies
│   ├── embeddings.py        # Gemini embedding wrapper
│   ├── vector_store.py      # ChromaDB operations
│   ├── rag_chain.py         # RAG pipeline
│   ├── chat_manager.py      # Conversation history
│   └── utils.py             # Helper utilities
│
├── ui/                       # Streamlit UI components
│   ├── sidebar.py           # Upload, settings, stats
│   ├── chat_page.py         # Chat interface
│   ├── vault_explorer.py    # Browse notes
│   └── analytics_page.py    # Statistics dashboard
│
├── config/                   # Configuration
│   └── settings.py          # App constants
│
├── assets/                   # Static assets
│   └── style.css            # Custom CSS
│
├── sample_vault/             # Demo Obsidian vault
│   ├── Machine Learning.md
│   ├── Neural Networks.md
│   ├── Transformer Architecture.md
│   ├── Python Tips.md
│   └── Daily Notes/
│       ├── 2026-08-01.md
│       └── 2026-08-15.md
│
└── tests/                    # Tests
```

## 🔧 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Frontend** | Streamlit | Free hosting, rapid UI, interactive |
| **LLM** | Google Gemini 1.5 Flash | Free tier, fast, great for RAG |
| **Embeddings** | Gemini text-embedding-004 | Free, 768 dimensions |
| **Vector DB** | ChromaDB | Simple, in-process, no setup |
| **Framework** | LangChain | Clean RAG abstractions |
| **Chunking** | RecursiveCharacterTextSplitter | Markdown-aware splitting |

## 🧪 How RAG Works

1. **Upload** → Your markdown files are parsed, extracting text, tags, and links
2. **Chunk** → Documents are split into overlapping chunks (500 tokens, 50 overlap)
3. **Embed** → Chunks are converted to vector embeddings via Gemini
4. **Index** → Vectors are stored in ChromaDB for fast similarity search
5. **Query** → Your question is embedded, and the top-K most relevant chunks are retrieved
6. **Generate** → Retrieved chunks are fed to Gemini as context to generate an answer
7. **Cite** → The answer includes references to which source notes provided the information

## 📝 Sample Questions

| Question | Expected Behavior |
|----------|-------------------|
| "What are my notes about?" | General overview of vault topics |
| "Explain transformers" | Answer from Transformer Architecture.md with citation |
| "What did I write on August 1st?" | Retrieve from daily note 2026-08-01.md |
| "Compare neural networks and transformers" | Cross-reference multiple notes |
| "What is quantum computing?" | "I couldn't find this in your notes" (not in vault) |

## 📜 License

MIT License

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Google AI](https://ai.google.dev/) for free Gemini API access
- [LangChain](https://langchain.com/) for RAG pipeline tools
- [ChromaDB](https://www.trychroma.com/) for vector storage
