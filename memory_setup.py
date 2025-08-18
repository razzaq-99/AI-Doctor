from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

data_path = "data/"

def load_pdf(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents
documents = load_pdf(data_path)




def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(documents)



def get_embedding_model():
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return model

embedding_model = get_embedding_model()


db_path = 'vectorstore/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(db_path)
