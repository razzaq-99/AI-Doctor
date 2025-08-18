from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
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


def set_prompt(prompt_template):
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

prompt = set_prompt(prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(huggingface_repo_id),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {"k": 3}),
    return_source_documents = True ,
    chain_type_kwargs = {
        'prompt': prompt
    }
)


user_query = input("Enter your question: ")
response = qa_chain.invoke({"query": user_query})

print("RESULT : ", response['result'])
print("SOURCE DOCUMENTS : ", response['source_documents'])