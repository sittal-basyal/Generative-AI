from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

PATH = 'Data/attention.pdf'

# Data ingestion
loader = PyPDFLoader(PATH)
docs = loader.load()

# Text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500 , 
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

# Embeddings 
embedding_model = HuggingFaceEmbeddings( 
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    )

# VectorStoreDB
vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory='chroma-db',
    collection_name='attention_paper')