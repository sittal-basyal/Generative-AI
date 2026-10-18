from httpx2 import query
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai"
)

search_tool = TavilySearch(max_results = 5 )

prompt = ChatPromptTemplate.from_template(
    ''' 
You are a helpful assistant that summarizes the given news into clear bullet points 
{news}
'''
)

parser = StrOutputParser()

chain = prompt | llm |parser

query = "latest car news of 2026 about flying car "
news_result = search_tool.run(query)

result = chain.invoke({'news':news_result})

print(result)
