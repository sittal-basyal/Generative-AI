#facebook ai similarity search 
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS , Chroma
from rich import print

from dotenv import load_dotenv
load_dotenv()

# Data Ingestion 
PATH = 'Data/attention.pdf'
loader = PyPDFLoader(PATH)
docs = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300 , chunk_overlap = 20
)
chunks = splitter.split_documents(docs)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# embeddings = embedding_model.embed_documents(chunks)"""


# embedding_model = OllamaEmbeddings(model = 'nomic-embed-text') # bydefault uses lamma2

# vector database

db = FAISS.from_documents(chunks, embedding_model)
query = "What is encoder and decoder ? "

# retrieved_docs = db.similarity_search(query)

# print(retrieved_docs[0].page_content)


# using retriever 

retriever = db.as_retriever()
final_doc = retriever.invoke(query)

print(final_doc[0].page_content)

db.save_local('faiss_db')