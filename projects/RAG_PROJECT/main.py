from typing import Collection

from langchain.chat_models import init_chat_model
from langchain_community import retrievers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

vector_store = Chroma(
    persist_directory='chroma-db',
    embedding_function= embedding_model,
    collection_name='attention_paper'
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k':4,
                     'fetch_k':10,
                     'lambda_mult':0.5
                       }
)


llm = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai"
)

#prompt template


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


print("*"*10 ,"RAG System " ,'*'*10)
while True:
    print('**********Press 0 to exit ...***********')
    query = input("You : ")
    if query == '0':
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs ]
    )

    final_prompt = prompt.invoke({
        'context':context,
        'question':query
        }
    )

    response = llm.invoke(final_prompt)
    print("\n")
    print("AI :",response.content)
