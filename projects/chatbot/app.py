import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Sittal's AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

model = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai",
    temperature=0.9
)

st.title("🤖 Sittal's AI Chatbot")
st.caption("Choose a personality and start chatting!")

mode_choice = st.radio(
    "Choose your AI personality:",
    ["😡 Angry", "😂 Funny", "😢 Sad"],
    horizontal=True
)

if mode_choice == "😡 Angry":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif mode_choice == "😂 Funny":
    mode = "You are a very funny AI agent. You respond with humor and jokes."
else:
    mode = "You are a very sad AI agent. You respond in a sad and emotional tone."

if "messages" not in st.session_state or st.session_state.get("current_mode") != mode:
    st.session_state.current_mode = mode
    st.session_state.messages = [SystemMessage(content=mode)]

for message in st.session_state.messages[1:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

prompt = st.chat_input("Type your message...")

if prompt:
    if prompt == "0":
        st.session_state.messages = [SystemMessage(content=mode)]
        st.rerun()

    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))

st.divider()

if st.button("🔄 Reset Chat", use_container_width=True):
    st.session_state.messages = [SystemMessage(content=mode)]
    st.rerun()