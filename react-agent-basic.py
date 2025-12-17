from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.2,
    base_url="http://localhost:11434"
)

# response = llm.invoke([
#     SystemMessage(content="You are a helpful AI assistant."),
#     HumanMessage(content="Explain transformers in simple terms.")
# ])

# print(response.content)


print(llm.invoke("Give me tweet about today's weather in bangalore."))



