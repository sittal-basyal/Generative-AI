from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich import print 

from dotenv import load_dotenv
load_dotenv()


# data ingestion 

web_path = 'https://docs.langchain.com/oss/python/langgraph/overview'

loader = WebBaseLoader(web_path=web_path)
docs = loader.load()

# text-splitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
chunks = splitter.split_documents(docs)

# embeddings

embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# vectorstore

vector_store = Chroma.from_documents(
    documents=chunks,
    persist_directory='VectorDatabase/langGraph-overview',
    embedding = embedding_model,
    collection_name = "langGraph-overview"
)