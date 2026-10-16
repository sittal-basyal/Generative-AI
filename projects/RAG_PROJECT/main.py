from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

model = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai"
)

PATH = 'Data/attention.pdf'

# data ingestion 
loader = PyPDFLoader(PATH)
docs = loader.load()

# data transformation - text chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size =300 ,
                                               chunk_overlap = 20)
chunks = text_splitter.split_documents(docs)
# print(chunks)

# embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vector_database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="attention_paper"
)

#query
query = """ The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
[20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building
block, computing hidden representations in parallel for all input and output positions. """
response = model.invoke("Hi")

retrieved_results = db.similarity_search(query=query)

