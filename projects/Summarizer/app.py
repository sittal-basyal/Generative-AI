
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# Model
model = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai"
)


# Movie Schema
class Movie(BaseModel):
    title: str
    release_year: int | None = None
    genre: List[str]
    director: str | None = None
    cast: List[str]
    rating: float | None = None
    Summary: str


structured_model = model.with_structured_output(Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Extract movie information from the paragraph.
    Use only information supported by the paragraph.
    Do not invent missing details.
    """),
    ("human", "{paragraph}")
])

chain = prompt | structured_model


# Streamlit UI
st.title("🎬 Movie Information Extractor")

user = st.text_area(
    "Enter a movie paragraph:",
    placeholder="Enter movie information here..."
)

if st.button("Extract Movie Information"):
    if user.strip():
        with st.spinner("Extracting movie information..."):
            result = chain.invoke({
                "paragraph": user
            })

        st.subheader("Extracted Movie Details")

        st.write("**Title:**", result.title)
        st.write("**Release Year:**", result.release_year)
        st.write("**Genre:**", ", ".join(result.genre))
        st.write("**Director:**", result.director)
        st.write("**Cast:**", ", ".join(result.cast))
        st.write("**Rating:**", result.rating)
        st.write("**Summary:**", result.Summary)
    else:
        st.warning("Please enter a movie paragraph.")