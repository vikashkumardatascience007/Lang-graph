from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
#from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.2:latest")

class BasicChatState(TypedDict):
    messages:Annotated[list, add_messages]
    

def chatbot(state:BasicChatState):
    return {
        "messages":[llm.invoke(state['messages'])]
    }
    
graph=StateGraph(BasicChatState)
graph.add_node('chatbot',chatbot)
graph.add_edge('chatbot',END)
graph.set_entry_point("chatbot")

app = graph.compile()

while True:
    
    user_input=input('User: ')
    if(user_input in ["exit", "end"]):
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(result)
        
        
    


    
    