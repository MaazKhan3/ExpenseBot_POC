# graph.py

from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from typing import TypedDict, Optional, List, Dict, Any

from app.intelligent_agent_v3.tools import (
    IntentTool,
    ExtractExpenseTool,
    CreateExpenseTool,
    GenerateSQLTool,
    ExecuteSQLTool,
    FormatBreakdownTool
)


# Define the global state shared across nodes
class AgentState(TypedDict):
    message: str
    phone_number: str
    intent: Optional[str]
    db_user: Optional[Any]
    expenses: Optional[List[Dict]]
    sql: Optional[str]
    sql_result: Optional[str]
    final_response: Optional[str]


# Define a simple router function to branch based on intent
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "log_expense":
        return "extract_expense"
    elif intent == "query":
        return "generate_sql"
    elif intent == "breakdown":
        return "format_breakdown"
    else:
        return "fallback"


def build_agent_graph() -> Runnable:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("detect_intent", IntentTool())
    graph.add_node("extract_expense", ExtractExpenseTool())
    graph.add_node("create_expense", CreateExpenseTool())
    graph.add_node("generate_sql", GenerateSQLTool())
    graph.add_node("execute_sql", ExecuteSQLTool())
    graph.add_node("format_breakdown", FormatBreakdownTool())

    # Fallback for unsupported intent
    graph.add_node("fallback", lambda state: {
        **state,
        "final_response": "Sorry, I couldn't understand that. Please try again."
    })

    # Response terminal
    graph.add_node("respond", lambda state: state)

    # Edges
    graph.set_entry_point("detect_intent")
    graph.add_conditional_edges("detect_intent", route_by_intent)

    graph.add_edge("extract_expense", "create_expense")
    graph.add_edge("create_expense", "respond")

    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "respond")

    graph.add_edge("format_breakdown", "respond")
    graph.add_edge("fallback", "respond")

    graph.set_finish_point("respond")

    return graph.compile()


# Instantiate agent_executor globally
agent_executor = build_agent_graph()