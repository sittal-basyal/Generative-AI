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


parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ('system','You are a code generator '),
    ('human','{topic}')
])

explain_prompt = ChatPromptTemplate.from_messages([
    ('system','you are an programming expert that explains the code in few words '),
    ('human',"Explain the following code line by line \n {code}")
])

seq1   = code_prompt | llm | parser 
seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explaination": explain_prompt | llm | parser
}
)

chain = seq1 | seq2

result = chain.invoke("write a code printing 'hello world' in python ")
print(result['code'])
print(result['explaination'])