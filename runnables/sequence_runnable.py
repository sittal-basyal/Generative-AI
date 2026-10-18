from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnableParallel , RunnableLambda , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
        model="ministral-3b-latest",
        model_provider="mistralai"
    )



output_parser = StrOutputParser()

# step by step manually 
'''
formatted_prompt = prompt.from_template(
    topic = 'Machine Learning'
)

response = llm.invoke(formatted_prompt)

final_output = output_parser.parse(response)
print(final_output)
'''


# SEQUENCE RUNNABLE
'''
prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in simple words '
)

chain = prompt | llm | output_parser

final_output = chain.invoke({'topic': 'Machine Learning'})

print(final_output)
'''

# PARALLEL RUNNABLE

short_prompt =  ChatPromptTemplate.from_template(
    'Explain {topic1} and {topic3} in 2 lines '
)

long_prompt = ChatPromptTemplate.from_template(
    'Explain {topic2} in 5 lines '
)



chain = RunnableParallel({
'short':RunnableLambda(lambda x : x['shorty']) |  short_prompt | llm | output_parser,
'long': RunnableLambda(lambda x : x['long']) | long_prompt | llm | output_parser
})

final_result = chain.invoke({
    "shorty": {"topic1" : "Machine Learning" ,
                         "topic3" : 'Deep learning'} ,
    "long": {"topic2":"neural Networks"}
                         })

print(final_result['short'])
print(final_result['long'])

