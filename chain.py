from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv


load_dotenv()


llm = ChatOllama(
    model="gemma:2b",   
    temperature=0.5
)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Load FAISS DB ----------------
db_path = 'vectorstore/db_faiss'
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


# ---------------- Prompt ----------------
prompt_template = """
Use the pieces of information provided in the context to answer the user's questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context : {context}
Question : {question}

Start the answer directly, No small talks please.
"""

def set_prompt(prompt_template):
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

prompt = set_prompt(prompt_template)


# ---------------- RetrievalQA ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)


# ---------------- Ask user ----------------
user_query = input("Enter your question: ")

response = qa_chain.invoke({"query": user_query})

print("\n=== LangChain QA ===")
print("RESULT : ", response['result'])
print("SOURCE DOCUMENTS : ", response['source_documents'])
