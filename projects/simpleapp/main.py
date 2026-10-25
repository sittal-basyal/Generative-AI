from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich import print 

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model= "ministral-3b-latest",
    model_provider="mistralai"
)

# embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# vectorstore

vector_store = Chroma(
    embedding_function = embedding_model,
    collection_name = "langGraph-overview",
    persist_directory='VectorDatabase/langGraph-overview'
)

# retrieve 

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs={
        'k':4,
        'fetch_k':10
    }
)

# # Test retrieval
# results = retriever.invoke("What is LangGraph?")

# for doc in results:
#     print(doc.page_content)