from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from rich import print 
load_dotenv()
#creating a tool

@tool
def get_text_length(text : str)-> int:
    ''' Returns the length of the text'''
    return len(text)

llm = init_chat_model(
    model = 'ministral-3b-latest',
    model_provider= 'mistralai'
)

# tool binding

llm_with_tool = llm.bind_tools(tools=[get_text_length])

# llm suggests the tool but not execute it 
result = llm_with_tool.invoke(" Give me the number of characters present in the text : 'Hello World!'")
# print(result)   
# print(result.tool_calls[0])

# tool execution

if result.tool_calls:

    tool_call = result.tool_calls[0]

    tool_result = get_text_length.invoke(tool_call['args'])

    final_result = llm.invoke(f"The length of text is : {tool_result}")

    print(final_result.content)