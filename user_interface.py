import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  

# ---------------- Loading data from vectorstore ----------------
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
        num_predict=512
    )
    
    
# ---------------- Main method ----------------
def main():
    st.title("Ask Chatbot")

    with st.sidebar:
        st.title("AI DOCTOR")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

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
            st.session_state.messages.append({"role": "assistant", "content": result})

            with st.expander("Source documents"):
                if not source_documents:
                    st.write("No source documents returned.")
                else:
                    for i, doc in enumerate(source_documents, 1):
                        meta = getattr(doc, "metadata", {})
                        src = meta.get("source") or meta.get("file_path") or "Unknown source"
                        st.markdown(f"**{i}. {src}**")
                        st.code((doc.page_content or "")[:2000])  
                        
                        
        # ---------------- Handling errors and exceptions ----------------
        except Exception as e:
            msg = str(e)
            if "assert d == self.d" in msg:
                st.error(
                    "Embedding dimension mismatch detected. "
                    "Your FAISS index was likely built with a different embedding model. "
                    f"Rebuild the index using `{EMBED_MODEL_NAME}`."
                )
            elif "Connection refused" in msg or "Failed to establish a new connection" in msg:
                st.error(
                    f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running and the model is pulled:\n\n"
                    f"  ollama pull {OLLAMA_MODEL}\n"
                    "  ollama serve"
                )
            else:
                st.error(f"Error: {msg}")

if __name__ == "__main__":
    main()
