from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

from chains import generation_chain, reflection_chain

load_dotenv()

# -----------------------------
# 1. State Definition
# -----------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]

# -----------------------------
# 2. Initialize Graph
# -----------------------------
graph = StateGraph(AgentState)

GENERATE = "generate"
REFLECT = "reflect"

# -----------------------------
# 3. Nodes
# -----------------------------
def generate_node(state: AgentState) -> AgentState:
    response = generation_chain.invoke({
        "messages": state["messages"]
    })
    return {
        "messages": state["messages"] + [response]
    }

def reflect_node(state: AgentState) -> AgentState:
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {
        "messages": state["messages"] + [
            HumanMessage(content=response.content)
        ]
    }

# -----------------------------
# 4. Add Nodes
# -----------------------------
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

# -----------------------------
# 5. Conditional Logic
# -----------------------------
def should_continue(state: AgentState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

graph.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        REFLECT: REFLECT,
        END: END
    }
)

graph.add_edge(REFLECT, GENERATE)

# -----------------------------
# 6. Compile
# -----------------------------
app = graph.compile()

# -----------------------------
# 7. DISPLAY FULL GRAPH
# -----------------------------
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# -----------------------------
# 8. Run
# -----------------------------
final_state = app.invoke(
    {
        "messages": [
            HumanMessage(content="AI Agents taking over content creation")
        ]
    }
)

print("\n--- FINAL OUTPUT ---\n")
for msg in final_state["messages"]:
    print(f"{msg.type.upper()}: {msg.content}\n")
