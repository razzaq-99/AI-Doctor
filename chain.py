from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_hubbingface import HuggingFaceEndpoint
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        model_kwargs = {
            "token":HF_TOKEN,
            "max_length": 512
        }
    )
    return llm 



db_path = 'vectorstore/db_faiss'
prompt_template = """"
Use the pieces of information provided in the context to answer the user's questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context : {context}
Question : {question}

Start the answer directly, No small talks please.
"""
