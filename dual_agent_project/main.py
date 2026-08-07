import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env
load_dotenv()

# Initialize LLM using NVIDIA API
llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",  # Popular high-performance model on NVIDIA API
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2
)

# Shared Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    iteration_count: int

# --- AGENT 1: DEVELOPER AGENT ---
def developer_node(state: AgentState) -> dict:
    """Agent 1: Writes code and documentation, or corrects it based on feedback."""
    system_prompt = SystemMessage(
        content=(
            "You are an Expert Software Engineer and Technical Writer.\n"
            "Your job is to provide clean, working code and concise documentation.\n"
            "If the Reviewer Agent provided feedback on previous errors, fix all issues carefully."
        )
    )
    
    response = llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

# --- AGENT 2: REVIEWER AGENT ---
def reviewer_node(state: AgentState) -> dict:
    """Agent 2: Evaluates code for syntax errors, logical bugs, and edge cases."""
    system_prompt = SystemMessage(
        content=(
            "You are a Senior Code Reviewer and QA Engineer.\n"
            "Evaluate the Developer's latest code and documentation.\n"
            "Rule 1: If there are ANY bugs, missing documentation, or syntax errors, state the issues clearly.\n"
            "Rule 2: If the code is correct, functional, and well-documented, respond ONLY with the exact text: 'APPROVED'."
        )
    )
    
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# --- ROUTER LOGIC ---
def route_after_review(state: AgentState) -> str:
    """Decides whether to cycle back to Developer or terminate."""
    last_message = state["messages"][-1].content.strip()
    iterations = state.get("iteration_count", 0)
    
    # Circuit breaker: stop after 3 failed revision loops to save API credits
    if iterations >= 3:
        print("\n[System]: Max revisions reached. Stopping workflow.")
        return END

    if "APPROVED" in last_message:
        return END
    
    print("\n[System]: Reviewer found issues. Sending back to Developer for fixes...")
    return "developer"

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("developer", developer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "developer": "developer",
        END: END
    }
)

app = workflow.compile()

# --- RUNNING THE SYSTEM ---
if __name__ == "__main__":
    user_prompt = input("Enter your coding task: ")
    
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "iteration_count": 0
    }
    
    print("\n--- Starting Autonomous Dual-Agent Workflow (Powered by NVIDIA) ---\n")
    
    for event in app.stream(initial_state, stream_mode="values"):
        latest_msg = event["messages"][-1]
        
        if isinstance(latest_msg, AIMessage):
            print("=== Response from Agent ===")
            print(latest_msg.content)
            print("-" * 50)