from langchain import messages
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = init_chat_model(
    model="ministral-3b-latest",
    model_provider='mistralai'
)

#chathistory- short term memory 
messages = []

''
# Ask for a prompt
print(f"{"-"*10}Welcome to Sittal's Chatbot Type 0 to exit{"-"*10}")
while True:
    
    prompt = input("Prompt: ")
    if prompt == 0:
        break
    messages.append(prompt)
    response = model.invoke(messages)
    messages.append(response.content)
    print("Bot: ",response.content)

print(messages)

# This has many problems such as :
'''
No role seperation 
Just raw things 
Memory keps growing infinetly
Api cost increases overtime 
No control over contect window 
'''

# Role seperation 
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
    
messages = [
    SystemMessage(content = "You are a funny AI agent .")
]


# Ask for a prompt
print(f"{"-"*10}Welcome to Sittal's Chatbot Type 0 to exit{"-"*10}")
while True:
    
    prompt = input("Prompt: ")
    if prompt == 0:
        break
    messages.append(HumanMessage(content = prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: ",response.content)

print(messages)


