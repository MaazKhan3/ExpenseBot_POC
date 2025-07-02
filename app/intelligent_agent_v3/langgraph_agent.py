# langgraph_agent.py

from typing import TypedDict, Optional, List, Any, Dict
from langgraph.graph import END, StateGraph
from .tools import (
    IntentTool,
    ExtractExpenseTool,
    CreateExpenseTool,
    GenerateSQLTool,
    ExecuteSQLTool,
    FormatBreakdownTool,
    ChitchatTool,
    RespondTool,
)


# 1. Define LangGraph Agent State
class AgentState(TypedDict):
    phone_number: str
    message: str
    intent: Optional[str]
    db_user: Optional[dict]
    expenses: Optional[List[dict]]
    query: Optional[str]
    sql: Optional[str]
    sql_result: Optional[str]
    final_response: Optional[str]
    db: Optional[Any]
    pending_context: Optional[Dict[str, Any]]


# 2. Define Router
def router(state: AgentState):
    intent = state.get("intent")
    if intent == "log_expense":
        return "extract_expense"
    elif intent == "query":
        return "generate_sql"
    elif intent == "breakdown":
        return "generate_breakdown"
    elif intent == "chitchat":
        return "chitchat"
    else:
        return "fallback"


# 3. Build LangGraph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("detect_intent", IntentTool())
builder.add_node("extract_expense", ExtractExpenseTool())
builder.add_node("create_expense", CreateExpenseTool())
builder.add_node("generate_sql", GenerateSQLTool())
builder.add_node("execute_sql", ExecuteSQLTool())
builder.add_node("generate_breakdown", FormatBreakdownTool())
builder.add_node("chitchat", ChitchatTool())
builder.add_node("fallback", RespondTool())
builder.add_node("respond", RespondTool())

# Set edges
builder.set_entry_point("detect_intent")
builder.add_conditional_edges("detect_intent", router)
builder.add_edge("extract_expense", "create_expense")
builder.add_edge("create_expense", "respond")
builder.add_edge("generate_sql", "execute_sql")
builder.add_edge("execute_sql", "respond")
builder.add_edge("generate_breakdown", "respond")
builder.add_edge("chitchat", "respond")
builder.add_edge("fallback", "respond")
builder.add_edge("respond", END)

# Compile
expense_agent_graph = builder.compile()


# 4. Entry point with corrected config
def run_expense_agent(phone_number: str, message: str, db):
    state = {
        "phone_number": phone_number,
        "message": message,
        "intent": None,
        "db_user": None,
        "expenses": None,
        "query": None,
        "sql": None,
        "sql_result": None,
        "final_response": None,
        "db": db,
        "pending_context": None,
    }
    
    print("[DEBUG] Running agent with db:", db)
    result = expense_agent_graph.invoke(state)
    return result
