
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

st.set_page_config(
    page_title="Attention Paper AI",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_rag():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model,
        collection_name="attention_paper"
    )

    llm = init_chat_model(
        model="ministral-3b-latest",
        model_provider="mistralai"
    )

    return vector_store, llm


vector_store, llm = load_rag()

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Retrieval Configuration")

    k = st.slider("Retrieved documents", 1, 10, 4)
    fetch_k = st.slider("Candidate documents", 5, 20, 10)
    lambda_mult = st.slider("MMR diversity", 0.0, 1.0, 0.5, 0.1)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Knowledge Source**")
    st.caption("Attention Is All You Need.pdf")
    st.caption("Vector database: ChromaDB")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": max(k, fetch_k),
        "lambda_mult": lambda_mult
    }
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant answering questions
about the paper Attention Is All You Need.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the document."

Do not use outside knowledge."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])

st.title("📚 Attention Paper AI")
st.markdown(
    "Ask questions about **Attention Is All You Need**. "
    "Answers are grounded in your PDF."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            docs = st.session_state.sources.get(i)

            if docs:
                with st.expander("📄 Retrieved document chunks"):
                    for j, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {j}**")
                        st.caption(f"Page: {doc.metadata.get('page', 'Unknown')}")
                        st.write(doc.page_content)
                        st.divider()

query = st.chat_input("Ask something about the paper...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the paper and generating an answer..."):
            try:
                docs = retriever.invoke(query)

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

                assistant_index = len(st.session_state.messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state.sources[assistant_index] = docs

                with st.expander("📄 Retrieved document chunks"):
                    for j, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {j}**")
                        st.caption(
                            f"Page: {doc.metadata.get('page', 'Unknown')}"
                        )
                        st.write(doc.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Something went wrong: {e}")
