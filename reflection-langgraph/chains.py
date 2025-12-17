from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

# LLM
llm = ChatOllama(
    model="mistral",
    temperature=0.7
)


# Generation prompt
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Twitter tech influencer assistant tasked with writing high-quality, viral tweets. "
            "Generate the best tweet possible for the user's request. "
            "If critique is provided, respond with an improved version."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Reflection prompt
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer reviewing a tweet. "
            "Provide detailed critique and improvement suggestions including tone, length, virality, and style."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chains
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
