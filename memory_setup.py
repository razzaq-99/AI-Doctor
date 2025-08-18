from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data_path = "data/"

def load_pdf(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf(data_path)
print("length : ", len(documents))