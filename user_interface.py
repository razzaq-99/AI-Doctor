import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Config / constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CONV_DIR = Path("conversations")
LAST_CONV_FILE = CONV_DIR / "last_conv.json"
INDEX_FILE = CONV_DIR / "index.json"
CONV_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Loading data from vectorstore (cached) ----------------
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# ---------------- wrapping custom prompt from chain file ----------------
def set_custom_prompt(custom_prompt_template: str):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


@st.cache_resource
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_predict=512,
    )


# ---------------- Conversation persistence helpers (with index + cache) ----------------
def _conv_path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def _index_path() -> Path:
    return INDEX_FILE


def load_index():
    """Load a lightweight index of conversations (id, title, created_at, preview).
    This avoids reading every full conversation file on each rerun and improves sidebar performance.
    """
    p = _index_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")) or []
    except Exception:
        return []


def save_index(index: list):
    try:
        _index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def update_index_on_save(conv: dict):
    index = load_index()
    preview = ""
    try:
        if conv.get("messages"):
            last = conv["messages"][-1]
            preview = (last.get("content") or "")[:200]
    except Exception:
        preview = ""

    entry = {
        "id": conv["id"],
        "title": conv.get("title", "Untitled"),
        "created_at": conv.get("created_at", ""),
        "preview": preview,
    }

    found = False
    for i, e in enumerate(index):
        if e.get("id") == conv["id"]:
            index[i] = entry
            found = True
            break
    if not found:
        index.append(entry)

    
    index.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    save_index(index)


def remove_from_index(conv_id: str):
    index = load_index()
    index = [e for e in index if e.get("id") != conv_id]
    save_index(index)




def save_conversation(conv: dict):
    path = _conv_path(conv["id"])
    path.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        LAST_CONV_FILE.write_text(json.dumps({"last": conv["id"]}), encoding="utf-8")
    except Exception:
        pass
    try:
        update_index_on_save(conv)
    except Exception:
        pass


def load_conversation(conv_id: str):
    """Load a full conversation. Uses session_state cache to avoid repeated disk reads.
    """
    if "conv_cache" in st.session_state and conv_id in st.session_state["conv_cache"]:
        return st.session_state["conv_cache"][conv_id]

    path = _conv_path(conv_id)
    if not path.exists():
        return None
    try:
        conv = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if "conv_cache" not in st.session_state:
        st.session_state["conv_cache"] = {}
    st.session_state["conv_cache"][conv_id] = conv
    return conv


def create_conversation(title: str = None):
    now = datetime.utcnow().isoformat()
    conv_id = uuid.uuid4().hex
    title = title or f"Conversation - {now}"
    conv = {
        "id": conv_id,
        "title": title,
        "created_at": now,
        "messages": [],
    }
    save_conversation(conv)
    
    st.session_state["convs_index"] = load_index()
    return conv


def delete_conversation(conv_id: str):
    p = _conv_path(conv_id)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass

    try:
        remove_from_index(conv_id)
    except Exception:
        pass
    
    if LAST_CONV_FILE.exists():
        try:
            last = json.loads(LAST_CONV_FILE.read_text(encoding="utf-8"))
            if last.get("last") == conv_id:
                LAST_CONV_FILE.unlink()
        except Exception:
            pass
    
    if "conv_cache" in st.session_state and conv_id in st.session_state["conv_cache"]:
        del st.session_state["conv_cache"][conv_id]
    st.session_state["convs_index"] = load_index()


def rename_conversation(conv_id: str, new_title: str):
    conv = load_conversation(conv_id)
    if not conv:
        return None
    conv["title"] = new_title
    save_conversation(conv)
    
    st.session_state["convs_index"] = load_index()
    if "conv_cache" in st.session_state and conv_id in st.session_state["conv_cache"]:
        st.session_state["conv_cache"][conv_id] = conv
    return conv


def get_last_conversation_id():
    if LAST_CONV_FILE.exists():
        try:
            data = json.loads(LAST_CONV_FILE.read_text(encoding="utf-8"))
            return data.get("last")
        except Exception:
            return None
    return None


# ---------------- Main method ----------------
def main():
    st.set_page_config(page_title="AI Doctor", layout="wide")

    # ---------------- Some custom CSS ----------------
    st.markdown(
        """
        <style>
        /* Sidebar card & buttons */
        [data-testid="stSidebar"] nav {padding: 8px 8px 16px 8px}
        .sidebar-header {font-size:18px; font-weight:700; color: #ffffff; padding: 12px 8px; text-align:center}
        .new-chat {background-color:#2b2d31; color:#e6edf3; padding:8px 10px; border-radius:8px; margin:6px 4px; display:block; text-align:left}
        .conv-item {background-color:#2b2d31; color:#dfe6ee; padding:8px; border-radius:8px; margin:6px 4px; display:flex; align-items:center; justify-content:space-between}
        .conv-item:hover {background-color:#343541}
        .conv-title {overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:200px}
        .mini-btn {background:transparent; border:none; color:#9aa0a6; font-size:16px}
        .sidebar-footer {color:#9aa0a6; font-size:12px; padding:8px 8px}
        /* main area tweaks */
        .stChatMessage {max-width:100%;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    
    if "conv_cache" not in st.session_state:
        st.session_state["conv_cache"] = {}
    if "convs_index" not in st.session_state:
        st.session_state["convs_index"] = load_index()
    if "menu_conv_id" not in st.session_state:
        st.session_state["menu_conv_id"] = None
    if "menu_open" not in st.session_state:
        st.session_state["menu_open"] = False

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown('<div class="sidebar-header">AI Doctor</div>', unsafe_allow_html=True)

        if st.button("➕ New chat", key="new_conv_btn"):
            new_conv = create_conversation("New chat")
            st.session_state.current_conv_id = new_conv["id"]
            st.session_state.conversation_title = new_conv["title"]
            st.session_state.messages = new_conv["messages"]

        st.markdown("---")

        convs = st.session_state.get("convs_index", [])
        if convs:
            for c in convs:
                cid = c.get("id")
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    
                    if st.button(c.get("title", "Untitled"), key=f"open_{cid}", use_container_width=True):
                        loaded = load_conversation(cid)
                        if loaded:
                            st.session_state.current_conv_id = loaded["id"]
                            st.session_state.conversation_title = loaded.get("title", "")
                            st.session_state.messages = loaded.get("messages", [])
                            
                            st.session_state[f"msg_count_{cid}"] = min(50, max(1, len(st.session_state.messages)))

                with col2:
                    if st.button("⋯", key=f"menu_{cid}"):
                        # ---------------- Toggle menu ----------------
                        if st.session_state.get("menu_conv_id") == cid and st.session_state.get("menu_open"):
                            st.session_state["menu_open"] = False
                            st.session_state["menu_conv_id"] = None
                        else:
                            st.session_state["menu_open"] = True
                            st.session_state["menu_conv_id"] = cid

                
                if st.session_state.get("menu_open") and st.session_state.get("menu_conv_id") == cid:
                    new_name = st.text_input("Rename conversation", value=c.get("title", ""), key=f"rename_input_{cid}")
                    colr, cold = st.columns([0.7, 0.3])
                    with colr:
                        if st.button("Rename", key=f"rename_btn_{cid}"):
                            if new_name.strip():
                                renamed = rename_conversation(cid, new_name.strip())
                                if renamed and st.session_state.get("current_conv_id") == cid:
                                    st.session_state.conversation_title = renamed["title"]
                                st.session_state["menu_open"] = False
                                st.session_state["menu_conv_id"] = None
                    with cold:
                        if st.button("Delete", key=f"delete_btn_{cid}"):
                            delete_conversation(cid)
                            
                            if st.session_state.get("current_conv_id") == cid:
                                st.session_state.current_conv_id = None
                                st.session_state.conversation_title = ""
                                st.session_state.messages = []

                            st.session_state["menu_open"] = False
                            st.session_state["menu_conv_id"] = None

            st.markdown("---")

            
            st.write("**Current:**")
            st.write(st.session_state.get("conversation_title", "-"))

            if st.button("Export conversation (.txt)", key="export_btn"):
                cid = st.session_state.get("current_conv_id")
                if cid:
                    conv = load_conversation(cid)
                    if conv:
                        lines = []
                        for m in conv.get("messages", []):
                            role = m.get("role", "")
                            content = m.get("content", "")
                            lines.append(f"{role.upper()}:{content}")
                        text = "".join(lines)
                        fname = f"{conv.get('title','conv')[:30].replace(' ','_')}_{conv['id'][:8]}.txt"
                        st.download_button("Download TXT", data=text, file_name=fname, mime="text/plain")

            if st.button("Delete current conversation", key="delete_current"):
                cid = st.session_state.get("current_conv_id")
                if cid:
                    delete_conversation(cid)
                    st.session_state.current_conv_id = None
                    st.session_state.conversation_title = ""
                    st.session_state.messages = []

        else:
            st.info("No saved conversations yet. Click 'New chat' to start one.")

        st.markdown('<div class="sidebar-footer">Tip: Conversations are saved on disk in the <code>conversations/</code> folder.</div>', unsafe_allow_html=True)

    
    if "messages" not in st.session_state:
        last_id = get_last_conversation_id()
        if last_id:
            conv = load_conversation(last_id)
            if conv:
                st.session_state.current_conv_id = conv["id"]
                st.session_state.conversation_title = conv.get("title", "")
                st.session_state.messages = conv.get("messages", [])
                st.session_state[f"msg_count_{conv['id']}"] = min(50, max(1, len(st.session_state.messages)))
            else:
                new_conv = create_conversation("New chat")
                st.session_state.current_conv_id = new_conv["id"]
                st.session_state.conversation_title = new_conv["title"]
                st.session_state.messages = new_conv["messages"]
                st.session_state[f"msg_count_{new_conv['id']}"] = min(50, max(1, len(st.session_state.messages)))
        else:
            new_conv = create_conversation("New chat")
            st.session_state.current_conv_id = new_conv["id"]
            st.session_state.conversation_title = new_conv["title"]
            st.session_state.messages = new_conv["messages"]
            st.session_state[f"msg_count_{new_conv['id']}"] = min(50, max(1, len(st.session_state.messages)))

    
    st.subheader(st.session_state.get("conversation_title", "Conversation"))

    
    cid = st.session_state.get("current_conv_id")
    messages = st.session_state.get("messages", []) or []
    if cid:
        total = len(messages)
        key_count = f"msg_count_{cid}"
        if key_count not in st.session_state:
            st.session_state[key_count] = min(50, max(1, total))
        show_count = st.session_state[key_count]

        
        if total > show_count:
            if st.button(f"Load earlier messages ({total - show_count} more)", key=f"load_more_{cid}"):
                st.session_state[key_count] = min(total, show_count + 50)

        
        to_render = messages[-show_count:]
        for message in to_render:
            role = message.get("role", "user")
            st.chat_message(role).markdown(message.get("content", ""))
    else:
        st.info("No conversation selected.")

    
    prompt = st.chat_input("Pass your prompt here")
    if prompt:
        
        if not st.session_state.get("current_conv_id"):
            conv = create_conversation("New chat")
            st.session_state.current_conv_id = conv["id"]
            st.session_state.conversation_title = conv["title"]
            st.session_state.messages = conv["messages"]

        
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        st.chat_message("user").markdown(prompt)

        
        cid = st.session_state["current_conv_id"]
        conv = load_conversation(cid) or {"id": cid, "title": st.session_state.get("conversation_title", ""), "created_at": datetime.utcnow().isoformat(), "messages": []}
        conv["messages"] = st.session_state.messages
        save_conversation(conv)
        st.session_state.conversation_title = conv.get("title", st.session_state.get("conversation_title", ""))
        
        st.session_state["convs_index"] = load_index()

        # ---------------- Wrapping custom prompt ----------------
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            llm = get_llm()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            response = qa_chain.invoke({"query": prompt})
            result = response.get("result", "").strip()
            source_documents = response.get("source_documents", [])

            
            st.chat_message("assistant").markdown(result)
            assistant_msg = {"role": "assistant", "content": result}
            st.session_state.messages.append(assistant_msg)

            
            conv = load_conversation(cid) or conv
            conv["messages"] = st.session_state.messages
            
            conv["title"] = st.session_state.get("conversation_title", conv.get("title", "Conversation"))
            save_conversation(conv)
            st.session_state["convs_index"] = load_index()

            # ---------------- Source Documents ----------------
            with st.expander("Source documents"):
                if not source_documents:
                    st.write("No source documents returned.")
                else:
                    for i, doc in enumerate(source_documents, 1):
                        meta = getattr(doc, "metadata", {}) or {}
                        src = meta.get("source") or meta.get("file_path") or "Unknown source"
                        st.markdown(f"**{i}. {src}**")
                        st.code((doc.page_content or "")[:2000])

        except Exception as e:
            msg = str(e)
            if "assert d == self.d" in msg:
                st.error(
                    "Embedding dimension mismatch detected. "
                    "Your FAISS index was likely built with a different embedding model. "
                    f"Rebuild the index using {EMBED_MODEL_NAME}."
                )
            elif "Connection refused" in msg or "Failed to establish a new connection" in msg:
                st.error(
                    f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running and the model is pulled:"
                    f"  ollama pull {OLLAMA_MODEL}"
                    "  ollama serve"
                )
            else:
                st.error(f"Error: {msg}")


if __name__ == "__main__":
    main()
