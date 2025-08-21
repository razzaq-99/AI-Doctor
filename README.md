# AI Doctor — Your Ultimate Medical Guider

A privacy‑friendly **RAG (Retrieval‑Augmented Generation)** assistant that answers medical questions **only** from your **most authentic, trusted medical book** (your curated *core text*) and any additional **vetted** references you add. It runs locally via **Ollama** and shows the **exact source excerpts** used to generate every answer.

The chat UI is **ChatGPT‑style** with a clean sidebar, hover menus (**⋮**) for **Rename/Delete**, and **persistent conversations** that survive refreshes and IDE restarts.

> ⚠️ **Medical Disclaimer**  
> This project is for **educational/informational** use only and does **not** constitute medical advice. For diagnosis or treatment, consult a qualified clinician.

---


## Why This Project

Traditional chatbots often hallucinate or mix web data of uncertain quality. **AI Doctor** is designed for **evidence‑first** interactions:

- The assistant is **hard‑grounded** on your **trusted medical textbook** (e.g., a standard internal medicine text) and any other **peer‑reviewed** or **official** sources *you* add.  
- If the answer isn’t supported by the indexed content, it **abstains** (“I don’t know from this context”) rather than guess.  
- Every answer is accompanied by **traceable source snippets** so you can verify the evidence immediately.  
- Everything runs **locally** (Ollama + FAISS), giving you control and privacy.

You control the corpus. Start with your **most authentic medical book** as the single source of truth, then extend with additional PDFs/notes as needed.

---

## Key Features

- **RAG‑grounded answers** using FAISS + HuggingFace embeddings; no ungrounded web browsing.  
- **Trustworthy corpus**: built around a **single, authoritative medical textbook** (*core text*); add vetted documents anytime.  
- **Strict safety prompt**: the model does **not** invent facts outside the context.  
- **Local LLM via Ollama**: default model `gemma:2b` (configurable).  
- **ChatGPT‑like UI** (Streamlit):
  - Sidebar shows **chat titles only** (no timestamps).  
  - Hover a chat to reveal a **⋮** menu with **Rename** / **Delete**.  
  - **Persistent conversations** saved as JSON on disk.  
- **Source transparency**: expandable **Source documents** panel for every answer.  
- **Simple indexing**: one script converts your PDFs into a FAISS vectorstore.

---

**Flow**  
1) Your question is embedded and matched against **FAISS** to fetch the most relevant chunks from your **core text**.  
2) The selected context + your question is passed into the LLM with a **strict prompt** that forbids guessing outside the context.  
3) The model responds; the UI shows the answer and an expandable list of **source documents**.

---

## Tech Stack

- **Frontend**: Streamlit  
- **RAG**: LangChain (`RetrievalQA`, `FAISS`)  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (configurable)  
- **LLM Runtime**: Ollama (`gemma:2b` by default; easily swap to Llama/Qwen/etc.)  
- **Vector DB**: FAISS (local, on disk)  
- **Config**: `.env` via `python-dotenv`

---

## Project Structure

```
.
├─ conversations/                # Persistent chats (auto-created)
│  └─ last_conv.json
├─ vectorstore/
│  └─ db_faiss/                  # FAISS index (created by build step)
├─ data/                         # Put your core medical textbook PDF(s) here
├─ user_interface.py             # Streamlit app (ChatGPT-like sidebar + RAG)
├─ build_vectorstore.py          # Indexer (PDF → chunks → FAISS)
├─ requirements.txt
└─ .env                          # App configuration
```

---

## Prerequisites

- **Python** ≥ 3.9 (3.10 recommended)  
- **Ollama** installed locally and reachable at `http://localhost:11434`  
  ```bash
  ollama pull gemma:2b
  ollama serve
  ```
- Your **core medical textbook** PDF (and any other vetted files) placed in `./data/`.

---

## Quick Start

```bash
# 1) Clone & enter
git clone <your-repo-url>
cd <your-repo>

# 2) Create & activate a virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure environment
# If .env.example exists: cp .env.example .env
# Otherwise create .env as shown below

# 5) Build the vectorstore from your core text in ./data
python build_vectorstore.py

# 6) Start Ollama (if not already)
ollama serve

# 7) Run the app
streamlit run user_interface.py
```

---

## Configuration

Create a **`.env`** file in the project root:

```env
# LLM settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:2b

# Embeddings
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Vectorstore path
DB_FAISS_PATH=vectorstore/db_faiss
```

**Tunable parameters** (in `user_interface.py`):  
- `temperature=0.3` (lower = more conservative)  
- `num_predict=512` (max output tokens)  
- Retriever `k` in `vectorstore.as_retriever(search_kwargs={"k": 3})`  
- Safety prompt `CUSTOM_PROMPT_TEMPLATE` (kept strict deliberately)

---


> **Tip:** If you ever change the embedding model, delete `vectorstore/db_faiss/` and rebuild to avoid dimension mismatches.

---

## Running the App

```bash
streamlit run user_interface.py
```

Open the URL that Streamlit prints (usually `http://localhost:8501`).

---

## Using the Chat UI

- **New Chat**: Click **➕ New Chat** in the sidebar.  
- **Open**: Click a chat title in the sidebar to load it.  
- **Hover Menu**: Hover a chat to reveal the **⋮** button → **Rename** or **Delete**.  
- **Ask Questions**: Type your query; the model answers **only from the indexed corpus** (your core text + vetted docs).  
- **Verify**: Expand **Source documents** to see the exact supporting excerpts/filenames.

**Persistence**  
Conversations are saved as JSON in `./conversations/` and automatically reloaded on restart.

---

## Data Quality & Safety

- **Single source of truth**: Start with one **trusted, authentic medical textbook** as your base (*core text*).  
- **Abstention over speculation**: If the corpus doesn’t contain the answer, the assistant says it **doesn’t know**.  
- **Provenance**: Every answer links to source snippets for manual verification.  
- **Privacy**: Runs locally; no external API calls are required.

> For production or clinical contexts, incorporate corpus governance (edition/version control, update logs, licensing), SME validation, and red‑team testing.

---

## Troubleshooting

**1) `Embedding dimension mismatch` / `assert d == self.d`**  
Your FAISS index was built with a different embedding model.  
**Fix:** Delete `vectorstore/db_faiss/` and rebuild with your current `EMBED_MODEL_NAME`.

**2) `Connection refused` (Ollama)**  
Ollama isn’t running or the base URL is wrong.  
**Fix:**
```bash
ollama pull gemma:2b
ollama serve
```
Ensure `.env` has `OLLAMA_BASE_URL=http://localhost:11434`.

**3) “No source documents returned.”**  
Your query didn’t match indexed content, the index is empty, or chunking is too coarse.  
**Fix:** Confirm `./data` has content and rebuild. Try smaller chunk size (700–900), higher overlap (150–250), and increase retriever `k` to 4–5.

**4) Streamlit port conflicts**  
Run: `streamlit run user_interface.py --server.port 8502`.

**5) Windows path issues**  
Keep the repo path short (e.g., `C:\Projects\AI-Doctor`).

---

## License

**MIT** — use, modify, and distribute with attribution.

