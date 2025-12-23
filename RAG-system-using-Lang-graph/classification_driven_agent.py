from typing import TypedDict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

 
llm = ChatOllama(model="mistral")


class AgentState(TypedDict):
    messages: List[BaseMessage]
    on_topic: str


def question_classifier(state: AgentState) -> AgentState:
    question = state["messages"][-1].content

    prompt = f"""
Answer ONLY with Yes or No.

Is the following question related to Peak Performance Gym
(history, owner, hours, membership, classes, trainers, facilities)?

Question: {question}
"""

    result = llm.invoke(prompt)
    answer = result.content.strip().lower()

    state["on_topic"] = "yes" if answer.startswith("yes") else "no"
    return state


def answer_question(state: AgentState) -> AgentState:
    question = state["messages"][-1].content

    prompt = f"""
You are a helpful assistant for Peak Performance Gym.

Answer the question clearly and concisely.

Question: {question}
"""

    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state


def off_topic_response(state: AgentState) -> AgentState:
    state["messages"].append(
        AIMessage(
            content="Sorry, I can only answer questions about Peak Performance Gym."
        )
    )
    return state


def route_question(state: AgentState):
    if state["on_topic"] == "yes":
        return "answer"
    return "off_topic"

 
graph = StateGraph(AgentState)

graph.add_node("classify", question_classifier)
graph.add_node("answer", answer_question)
graph.add_node("off_topic", off_topic_response)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_question,
    {
        "answer": "answer",
        "off_topic": "off_topic",
    },
)

graph.add_edge("answer", END)
graph.add_edge("off_topic", END)

app = graph.compile()

 
try:
    png_bytes = app.get_graph().draw_mermaid_png()

    with open("langgraph_flow.png", "wb") as f:
        f.write(png_bytes)

    print("Graph image saved as langgraph_flow.png")

except Exception as e:
    print("Failed to generate graph image:", e)

 
if __name__ == "__main__":
    user_question = "Who is the owner and what are the timings?"

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_question)]
        }
    )

    print("\nFINAL ANSWER:\n")
    print(result["messages"][-1].content)
