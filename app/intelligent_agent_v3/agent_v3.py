# agent_v3.py

from app.intelligent_agent_v3.langgraph_agent import run_expense_agent
from sqlalchemy.orm import Session

def process_message_with_agent_v3(phone_number: str, message: str, db: Session) -> dict:
    """
    Main entry point for FastAPI webhook to use the LangGraph ReAct Agent.
    """
    try:
        result = run_expense_agent(phone_number, message, db)

        final_response = result.get("final_response") or result.get("sql_result") or "Sorry, I couldn't process that."
        return {
            "message": final_response,
            "intent": result.get("intent", "unknown"),
        }

    except Exception as e:
        print(f"[Agent V3] Error: {e}")
        return {
            "message": "An unexpected error occurred while processing your message.",
            "intent": "error"
        }