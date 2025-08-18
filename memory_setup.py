from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

data_path = "data/"

def load_pdf(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents
documents = load_pdf(data_path)
print("length : ", len(documents))



def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(documents)
print('length :', len(text_chunks))