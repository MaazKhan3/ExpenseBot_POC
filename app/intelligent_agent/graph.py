"""
Intelligent ExpenseBot Agent - Proper Tool-Based Implementation
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict
from app.intelligent_agent.config import config
from app.intelligent_agent.memory import memory
from app.intelligent_agent.tools import ExpenseTools
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pydantic import SecretStr
import os
import json
from app import crud

logger = logging.getLogger("expensebot.intelligent_agent.graph")

# --- State ---
class AgentState(TypedDict):
    phone_number: str
    user_message: str
    conversation_history: List[Dict[str, Any]]
    user_id: Optional[int]
    response: Optional[str]
    pending_expense: Optional[Dict[str, Any]]
    intent: Optional[str]
    amount: Optional[float]
    category: Optional[str]
    item: Optional[str]
    clarification: Optional[str]
    db: Any
    tool_result: Optional[Dict[str, Any]]
    tool_name: Optional[str]
    final_response: Optional[str]
    multiple_expenses: Optional[List[Dict[str, Any]]]

# --- LLM Router Node (First LLM Call) ---
def llm_router_node(state: AgentState) -> AgentState:
    """First LLM call: Analyze intent and decide which tool to use"""
    phone_number = state["phone_number"]
    user_message = state["user_message"]
    user_id = state["user_id"]
    pending_expense = state.get("pending_expense")
    conversation_history = state.get("conversation_history", [])

    # Build conversation context
    context_lines = []
    for turn in conversation_history[-3:]:
        context_lines.append(f"User: {turn.get('user', '')}")
        if turn.get('assistant'):
            context_lines.append(f"Assistant: {turn['assistant']}")
    conversation_context = "\n".join(context_lines)

    # Check pending expense context
    pending_info = ""
    if pending_expense:
        pending_info = f"\nPENDING EXPENSE CONTEXT: Amount={pending_expense.get('amount')}, Item={pending_expense.get('item')}, Category={pending_expense.get('category')}"

    prompt = f"""
You are an intelligent expense tracking assistant. Analyze the user message and decide which tool to use.

User ID: {user_id}
Message: "{user_message}"
Conversation Context:
{conversation_context}{pending_info}

Available Tools:
1. log_expense_tool - For logging single expenses (complete or incomplete)
2. log_multiple_expenses_tool - For logging multiple expenses in one message
3. query_expenses_tool - For querying expense data (breakdowns, top expenses, etc.)
4. get_total_expenses_tool - For getting total expenses for time periods
5. greeting_tool - For greetings
6. clarification_tool - For asking for more information

Think step by step:
1. What does the user want to do?
2. Are they responding to a previous question or providing context?
3. Do they have complete information for an expense?
4. Which tool should handle this request?

RULES:
- If user provides multiple expenses (e.g., "apples 500, carrots 40, bananas 200"), use log_multiple_expenses_tool
- If user provides single expense with amount + item/category, use log_expense_tool
- If user provides only amount or only item, use log_expense_tool (it will handle clarification)
- If user asks about expense breakdowns, top expenses, or specific analysis, use query_expenses_tool
- If user asks about total spending for time periods (today, week, month), use get_total_expenses_tool
- If user asks "how much have I spent on [category]?", use query_expenses_tool
- If user greets, use greeting_tool
- If user says thanks/thank you/okay/good/great/also/no/not, use greeting_tool (acknowledgment)
- If user provides context for previous conversation (like "yeah i bought a car"), use query_expenses_tool to acknowledge
- If user sends single letters or very short unclear messages (like "i", "a", "spent"), use clarification_tool
- If unclear, use clarification_tool

EXPENSE RECOGNITION PATTERNS:
- "spent X on Y" → amount=X, item=Y
- "bought Y" → item=Y (ask for amount)
- "X for Y" → amount=X, item=Y
- "Y for X" → amount=X, item=Y
- "X PKR on Y" → amount=X, item=Y
- "Y cost X" → amount=X, item=Y
- "X, Y, Z" → multiple items (need amounts)
- "X amount, Y amount, Z amount" → multiple expenses

Output ONLY a JSON object with:
- "reasoning": Your step-by-step reasoning
- "tool_name": The exact tool name to use
- "intent": The user's intent
- "extracted_data": Any data you can extract (amount, item, category, etc.)
- "multiple_expenses": Array of expenses if multiple found

EXAMPLES:

User: "600 pkr spent on juice"
{{
  "reasoning": "Single complete expense: juice for 600 PKR",
  "tool_name": "log_expense_tool",
  "intent": "log_expense",
  "extracted_data": {{
    "amount": 600,
    "item": "juice",
    "category": "food"
  }}
}}

User: "spent 1400 on sweets"
{{
  "reasoning": "Single complete expense: sweets for 1400 PKR",
  "tool_name": "log_expense_tool",
  "intent": "log_expense",
  "extracted_data": {{
    "amount": 1400,
    "item": "sweets",
    "category": "food"
  }}
}}

User: "spent 900"
{{
  "reasoning": "Single incomplete expense: amount provided but item missing",
  "tool_name": "log_expense_tool",
  "intent": "log_expense",
  "extracted_data": {{
    "amount": 900
  }}
}}

User: "bought sweets"
{{
  "reasoning": "Single incomplete expense: item provided but amount missing",
  "tool_name": "log_expense_tool",
  "intent": "log_expense",
  "extracted_data": {{
    "item": "sweets",
    "category": "food"
  }}
}}

User: "apples 500, carrots 40, bananas 200"
{{
  "reasoning": "Multiple expenses provided: apples, carrots, bananas",
  "tool_name": "log_multiple_expenses_tool",
  "intent": "log_multiple_expenses",
  "multiple_expenses": [
    {{"amount": 500, "item": "apples", "category": "food"}},
    {{"amount": 40, "item": "carrots", "category": "food"}},
    {{"amount": 200, "item": "bananas", "category": "food"}}
  ]
}}

User: "fuel 500, hat 2k, watch 25k"
{{
  "reasoning": "Multiple expenses provided: fuel, hat, watch",
  "tool_name": "log_multiple_expenses_tool",
  "intent": "log_multiple_expenses",
  "multiple_expenses": [
    {{"amount": 500, "item": "fuel", "category": "transportation"}},
    {{"amount": 2000, "item": "hat", "category": "clothing"}},
    {{"amount": 25000, "item": "watch", "category": "electronics"}}
  ]
}}

User: "yeah i actually bought a car that day"
{{
  "reasoning": "User is providing context for previous conversation about expensive purchase",
  "tool_name": "query_expenses_tool",
  "intent": "provide_context",
  "extracted_data": {{
    "context": "car purchase",
    "item": "car"
  }}
}}

User: "show me my spending breakdown"
{{
  "reasoning": "User wants expense breakdown analysis",
  "tool_name": "query_expenses_tool",
  "intent": "query_expenses",
  "extracted_data": {{
    "query_type": "breakdown"
  }}
}}

User: "how much did I spend this week?"
{{
  "reasoning": "Query for weekly total expenses",
  "tool_name": "get_total_expenses_tool",
  "intent": "get_total_expenses",
  "extracted_data": {{
    "time_period": "week"
  }}
}}

User: "hi"
{{
  "reasoning": "Greeting",
  "tool_name": "greeting_tool",
  "intent": "greeting",
  "extracted_data": {{}}
}}

User: "thanks"
{{
  "reasoning": "User acknowledging/thankful",
  "tool_name": "greeting_tool",
  "intent": "acknowledgment",
  "extracted_data": {{}}
}}

User: "also"
{{
  "reasoning": "User acknowledging or wanting to add more",
  "tool_name": "greeting_tool",
  "intent": "acknowledgment",
  "extracted_data": {{}}
}}

User: "no not again"
{{
  "reasoning": "User expressing frustration or disagreement",
  "tool_name": "greeting_tool",
  "intent": "acknowledgment",
  "extracted_data": {{}}
}}

User: "i"
{{
  "reasoning": "Single letter message, unclear intent",
  "tool_name": "clarification_tool",
  "intent": "clarification",
  "extracted_data": {{}}
}}

User: "spent"
{{
  "reasoning": "Single word without context, unclear intent",
  "tool_name": "clarification_tool",
  "intent": "clarification",
  "extracted_data": {{}}
}}

User: "how much have I spent on food?"
{{
  "reasoning": "User asking for expense query about food category",
  "tool_name": "query_expenses_tool",
  "intent": "query_expenses",
  "extracted_data": {{
    "query_type": "category_spending",
    "category": "food"
  }}
}}

User: "how much have I spent on electronics?"
{{
  "reasoning": "User asking for expense query about electronics category",
  "tool_name": "query_expenses_tool",
  "intent": "query_expenses",
  "extracted_data": {{
    "query_type": "category_spending",
    "category": "electronics"
  }}
}}

Output ONLY the JSON object.
"""

    llm = ChatGroq(
        api_key=SecretStr(os.environ.get("GROQ_API_KEY") or ""),
        model=config.llm_model,
        temperature=0,
    )
    
    logger.info(f"🤖 Router LLM: Analyzing message: {user_message}")
    response = llm.invoke(prompt)
    llm_response = response.content if hasattr(response, "content") else str(response)
    
    if isinstance(llm_response, list):
        llm_response = " ".join(str(x) for x in llm_response)
    
    logger.info(f"🤖 Router LLM Response: {llm_response}")

    # Parse JSON response
    try:
        llm_response = llm_response.strip()
        first = llm_response.find('{')
        last = llm_response.rfind('}')
        
        if first != -1 and last != -1 and last > first:
            json_str = llm_response[first:last+1]
            action_data = json.loads(json_str)
            logger.info(f"🤖 Router parsed data: {action_data}")
        else:
            # Fallback
            action_data = {
                "reasoning": "Fallback analysis",
                "tool_name": "clarification_tool",
                "intent": "clarification",
                "extracted_data": {}
            }
    except Exception as e:
        logger.error(f"🤖 Router JSON parse error: {e}")
        action_data = {
            "reasoning": "JSON parse error fallback",
            "tool_name": "clarification_tool",
            "intent": "clarification",
            "extracted_data": {}
        }

    # Update state with router decision
    state["tool_name"] = action_data.get("tool_name", "clarification_tool")
    state["intent"] = action_data.get("intent", "clarification")
    
    # Extract data for tools
    extracted_data = action_data.get("extracted_data", {})
    state["amount"] = extracted_data.get("amount")
    state["item"] = extracted_data.get("item")
    state["category"] = extracted_data.get("category")
    
    # Extract multiple expenses if present
    multiple_expenses = action_data.get("multiple_expenses", [])
    state["multiple_expenses"] = multiple_expenses
    
    logger.info(f"🤖 Router decided: {state['tool_name']} for intent: {state['intent']}")
    if multiple_expenses:
        logger.info(f"🤖 Router extracted {len(multiple_expenses)} expenses: {multiple_expenses}")
    return state

# --- Tool Nodes ---
def log_expense_tool(state: AgentState) -> AgentState:
    """Tool: Handle expense logging with proper context awareness"""
    db = state.get("db")
    user_id = state.get("user_id")
    amount = state.get("amount")
    item = state.get("item")
    category = state.get("category")
    phone_number = state.get("phone_number")
    pending_expense = state.get("pending_expense")
    user_message = state.get("user_message")
    
    logger.info(f"🔧 Log Expense Tool: amount={amount}, item={item}, category={category}")
    logger.info(f"🔧 Pending expense: {pending_expense}")
    logger.info(f"🔧 User message: {user_message}")
    
    # Check if user_id is valid
    if user_id is None:
        state["tool_result"] = {
            "status": "error",
            "error": "User account not found",
            "response": "User account not found. Please try again."
        }
        return state
    
    # Handle context from pending expense
    if pending_expense and not amount:
        amount = pending_expense.get("amount")
        state["amount"] = amount
        logger.info(f"🔧 Using pending amount: {amount}")
    
    if pending_expense and not item:
        item = pending_expense.get("item")
        state["item"] = item
        logger.info(f"🔧 Using pending item: {item}")
    
    # Parse amount if string
    if amount and isinstance(amount, str):
        amount = parse_amount(amount)
        state["amount"] = amount
    
    # Determine what's missing
    missing_info = []
    if not amount:
        missing_info.append("amount")
    if not item:
        missing_info.append("item")
    
    # Check if this is a very short or unclear message that shouldn't use pending context
    if len(user_message.strip()) <= 3 or user_message.strip().lower() in ["i", "a", "e", "o", "u", "spent", "bought", "paid"]:
        # Clear pending context for unclear messages
        memory.set_pending_expense(phone_number, None)
        missing_info = ["amount", "item"]  # Force asking for both
    
    if missing_info:
        # Incomplete expense - ask for missing info
        if "amount" in missing_info and "item" in missing_info:
            response = "I need both the amount and what you bought. Could you please provide both? For example: '500 for popcorn' or 'popcorn 500'"
        elif "amount" in missing_info:
            response = f"What was the cost of {item}?"
        else:  # missing item
            response = f"What did you buy for {amount} PKR?"
        
        # Store pending expense context only if we have some valid data
        if amount or item:
            pending_data = {
                "amount": amount,
                "item": item,
                "category": category
            }
            memory.set_pending_expense(phone_number, pending_data)
        else:
            # Clear pending context if no valid data
            memory.set_pending_expense(phone_number, None)
        
        state["tool_result"] = {
            "status": "incomplete",
            "missing": missing_info,
            "response": response,
            "pending_expense": pending_data if (amount or item) else None
        }
    else:
        # Complete expense - log it
        try:
            # Auto-map category if not provided
            if not category and item:
                category = map_category(item)
                state["category"] = category
            
            # Ensure amount is a valid number
            if amount is None:
                raise ValueError("Amount is required")
            
            # Create category and expense
            db_category = crud.get_or_create_category(db, user_id, category or "misc")
            db_expense = crud.create_expense(
                db,
                user_id=user_id,
                category_id=getattr(db_category, "id"),
                amount=float(amount),
                note=item or ""
            )
            
            # Clear pending expense
            memory.set_pending_expense(phone_number, None)
            
            # Generate appropriate response based on context
            if pending_expense:
                response = f"Perfect! I've logged {amount:,.0f} PKR for {item} under {category}. Your expense has been saved successfully!"
            else:
                response = f"Great! I've logged {amount:,.0f} PKR for {item} under {category}. Your expense has been saved successfully!"
            
            state["tool_result"] = {
                "status": "success",
                "expense_id": getattr(db_expense, "id"),
                "amount": amount,
                "category": category or "misc",
                "item": item,
                "response": response
            }
            
            logger.info(f"✅ Logged expense: {amount} PKR for {category or item}")
            
        except Exception as e:
            logger.error(f"❌ Expense logging error: {e}")
            state["tool_result"] = {
                "status": "error",
                "error": str(e),
                "response": "Failed to log expense. Please try again."
            }
    
    return state

def query_expenses_tool(state: AgentState) -> AgentState:
    """Tool: Handle expense queries"""
    db = state.get("db")
    user_id = state.get("user_id")
    user_message = state.get("user_message")
    intent = state.get("intent")
    extracted_data = state.get("extracted_data", {})
    
    logger.info(f"🔧 Query Expenses Tool: {user_message}")
    
    # Check if user_id is valid
    if user_id is None:
        state["tool_result"] = {
            "status": "error",
            "error": "User account not found",
            "response": "User account not found. Please try again."
        }
        return state
    
    try:
        tools = ExpenseTools(db)
        
        # Handle context provision (like "yeah i bought a car")
        if intent == "provide_context":
            response = "Ah, that makes perfect sense! A car purchase would definitely be your biggest expense. Thanks for the context - that helps me understand your spending patterns better. Is there anything else you'd like to know about your expenses?"
            state["tool_result"] = {
                "status": "success",
                "response": response,
                "query_type": "context_acknowledgment"
            }
            return state
        
        # Handle category-specific spending queries
        if extracted_data.get("query_type") == "category_spending":
            category = extracted_data.get("category")
            if category:
                # Get expenses for specific category
                from sqlalchemy import text
                sql = text(f"""
                    SELECT COALESCE(SUM(e.amount), 0) as total, COUNT(*) as count
                    FROM expenses e
                    JOIN categories c ON e.category_id = c.id
                    WHERE e.user_id = :user_id AND LOWER(c.name) = LOWER(:category)
                """)
                
                result = db.execute(sql, {"user_id": user_id, "category": category}).fetchone()
                total = result[0] if result else 0
                count = result[1] if result else 0
                
                if total > 0:
                    response = f"You have spent a total of {total:,.0f} PKR on {category} across {count} transactions."
                else:
                    response = f"You haven't logged any expenses for {category} yet."
                
                state["tool_result"] = {
                    "status": "success",
                    "response": response,
                    "query_type": "category_spending",
                    "category": category,
                    "total": total,
                    "count": count
                }
                return state
        
        # Determine query type from message
        message_lower = user_message.lower()
        
        if "most expensive" in message_lower or "highest" in message_lower:
            result = tools.get_max_expense(user_id, "all")
            if result:
                response = f"Your most expensive purchase was {result['amount']:,.0f} PKR for {result['category']}."
            else:
                response = "I couldn't find any expenses in your records."
        
        elif "breakdown" in message_lower or "category" in message_lower:
            results = tools.get_category_breakdown(user_id, "all")
            if results:
                response = tools.format_expense_response(results, "category_breakdown")
            else:
                response = "I couldn't find any expenses to create a breakdown for."
        
        elif "top" in message_lower and ("3" in message_lower or "three" in message_lower):
            results = tools.get_top_expenses(user_id, 3, "all")
            if results:
                response = tools.format_expense_response(results, "top_expenses")
            else:
                response = "I couldn't find any expenses to show you."
        
        elif "top" in message_lower or "highest" in message_lower:
            results = tools.get_top_expenses(user_id, 5, "all")
            if results:
                response = tools.format_expense_response(results, "top_expenses")
            else:
                response = "I couldn't find any expenses to show you."
        
        else:
            # Default to category breakdown
            results = tools.get_category_breakdown(user_id, "all")
            if results:
                response = tools.format_expense_response(results, "category_breakdown")
            else:
                response = "I couldn't find any expenses in your records."
        
        state["tool_result"] = {
            "status": "success",
            "response": response,
            "query_type": "expense_analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        state["tool_result"] = {
            "status": "error",
            "error": str(e),
            "response": "Sorry, I couldn't retrieve your expense data right now."
        }
    
    return state

def get_total_expenses_tool(state: AgentState) -> AgentState:
    """Tool: Get total expenses for a time period"""
    db = state.get("db")
    user_id = state.get("user_id")
    user_message = state.get("user_message")
    
    logger.info(f"🔧 Get Total Expenses Tool: {user_message}")
    
    # Check if user_id is valid
    if user_id is None:
        state["tool_result"] = {
            "status": "error",
            "error": "User account not found",
            "response": "User account not found. Please try again."
        }
        return state
    
    try:
        from datetime import datetime
        from sqlalchemy import text
        
        # Determine time period
        message_lower = user_message.lower()
        if "yesterday" in message_lower:
            time_filter = "AND DATE(timestamp) = CURRENT_DATE - INTERVAL '1 day'"
            period = "yesterday"
        elif "week" in message_lower:
            time_filter = "AND timestamp >= NOW() - INTERVAL '7 days'"
            period = "this week"
        elif "month" in message_lower:
            time_filter = "AND timestamp >= NOW() - INTERVAL '1 month'"
            period = "this month"
        elif "today" in message_lower:
            time_filter = "AND DATE(timestamp) = CURRENT_DATE"
            period = "today"
        else:
            # Default to today
            time_filter = "AND DATE(timestamp) = CURRENT_DATE"
            period = "today"
        
        sql = text(f"""
            SELECT COALESCE(SUM(e.amount), 0) as total
            FROM expenses e
            WHERE e.user_id = :user_id {time_filter}
        """)
        
        result = db.execute(sql, {"user_id": user_id}).fetchone()
        total = result[0] if result else 0
        
        if total > 0:
            response = f"You have spent a total of {total:,.0f} PKR {period}."
        else:
            response = f"You haven't logged any expenses {period}."
        
        state["tool_result"] = {
            "status": "success",
            "total": total,
            "period": period,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"❌ Total expenses error: {e}")
        state["tool_result"] = {
            "status": "error",
            "error": str(e),
            "response": "Sorry, I couldn't retrieve your expenses right now."
        }
    
    return state

def greeting_tool(state: AgentState) -> AgentState:
    """Tool: Handle greetings and acknowledgments"""
    user_message = state.get("user_message", "").lower()
    intent = state.get("intent")
    
    # Handle acknowledgments and short responses
    if intent == "acknowledgment" or any(word in user_message for word in ["thanks", "thank you", "okay", "ok", "good", "great", "also", "no", "not"]):
        response = "You're welcome! Is there anything else I can help you with?"
    # Handle greetings
    elif any(word in user_message for word in ["hi", "hello", "good morning", "good afternoon", "good evening"]):
        response = "Hello! How can I help with your expenses today?"
    else:
        response = "Hello! How can I help with your expenses today?"
    
    state["tool_result"] = {
        "status": "success",
        "response": response,
        "greeting": True
    }
    return state

def clarification_tool(state: AgentState) -> AgentState:
    """Tool: Ask for clarification"""
    user_message = state.get("user_message", "").lower()
    
    # Handle very short responses and single letters
    if len(user_message.strip()) <= 2:
        response = "I didn't quite catch that. Could you please be more specific? For example:\n• '500 for groceries' to log an expense\n• 'How much did I spend this week?' to check expenses\n• 'Show me my spending breakdown' for analysis"
    elif user_message.strip() in ["i", "a", "e", "o", "u", "spent", "bought", "paid"]:
        response = "I need more information to help you. Could you please provide:\n• The amount you spent\n• What you bought\n\nFor example: '500 for groceries' or 'groceries 500'"
    else:
        response = "I'm not sure what you meant. You can:\n• Log expenses: '500 for groceries'\n• Ask queries: 'How much did I spend this week?'\n• Get breakdowns: 'Show me my spending breakdown'"
    
    state["tool_result"] = {
        "status": "clarification_needed",
        "response": response
    }
    return state

def log_multiple_expenses_tool(state: AgentState) -> AgentState:
    """Tool: Handle multiple expenses in a single message"""
    db = state.get("db")
    user_id = state.get("user_id")
    multiple_expenses = state.get("multiple_expenses", [])
    phone_number = state.get("phone_number")
    
    logger.info(f"🔧 Log Multiple Expenses Tool: {len(multiple_expenses) if multiple_expenses else 0} expenses")
    
    # Check if user_id is valid
    if user_id is None:
        state["tool_result"] = {
            "status": "error",
            "error": "User account not found",
            "response": "User account not found. Please try again."
        }
        return state
    
    if not multiple_expenses:
        state["tool_result"] = {
            "status": "error",
            "error": "No expenses to log",
            "response": "I couldn't find any expenses to log. Please try again."
        }
        return state
    
    try:
        logged_expenses = []
        failed_expenses = []
        
        for expense in multiple_expenses:
            amount = expense.get("amount")
            item = expense.get("item")
            category = expense.get("category")
            
            # Parse amount if string
            if amount and isinstance(amount, str):
                amount = parse_amount(amount)
            
            # Auto-map category if not provided
            if not category and item:
                category = map_category(item)
            
            # Validate expense data
            if not amount or not item:
                failed_expenses.append(expense)
                continue
            
            try:
                # Create category and expense
                db_category = crud.get_or_create_category(db, user_id, category or "misc")
                db_expense = crud.create_expense(
                    db,
                    user_id=user_id,
                    category_id=getattr(db_category, "id"),
                    amount=float(amount),
                    note=item or ""
                )
                logged_expenses.append({
                    "amount": amount,
                    "item": item,
                    "category": category or "misc"
                })
                logger.info(f"✅ Logged expense: {amount} PKR for {category or item}")
            except Exception as e:
                logger.error(f"❌ Failed to log expense {item}: {e}")
                failed_expenses.append(expense)
        
        # Generate response
        if logged_expenses and not failed_expenses:
            # All expenses logged successfully
            expense_list = [f"{exp['amount']:,.0f} PKR for {exp['item']}" for exp in logged_expenses]
            response = f"Perfect! I've logged {len(logged_expenses)} expenses: {', '.join(expense_list)}. All expenses have been saved successfully!"
        elif logged_expenses and failed_expenses:
            # Some expenses logged, some failed
            expense_list = [f"{exp['amount']:,.0f} PKR for {exp['item']}" for exp in logged_expenses]
            response = f"Great! I've logged {len(logged_expenses)} expenses: {', '.join(expense_list)}. Some expenses couldn't be logged due to missing information."
        else:
            # No expenses logged
            response = "I couldn't log any expenses. Please make sure to provide both amount and item for each expense."
        
        state["tool_result"] = {
            "status": "success",
            "logged_expenses": logged_expenses,
            "failed_expenses": failed_expenses,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"❌ Multiple expenses logging error: {e}")
        state["tool_result"] = {
            "status": "error",
            "error": str(e),
            "response": "Failed to log expenses. Please try again."
        }
    
    return state

# --- Final LLM Node (Second LLM Call) ---
def final_response_node(state: AgentState) -> AgentState:
    """Second LLM call: Generate final natural language response based on tool result"""
    tool_result = state.get("tool_result", {})
    user_message = state["user_message"]
    intent = state.get("intent")
    phone_number = state.get("phone_number")
    
    logger.info(f"🤖 Final Response LLM: Processing tool result: {tool_result}")
    
    # For successful expense logging, use the tool result and clear context
    if intent == "log_expense" and tool_result and tool_result.get("status") == "success":
        response = tool_result.get("response", "Expense logged successfully!")
        # Clear any pending expense context after successful logging
        memory.set_pending_expense(phone_number, None)
        state["final_response"] = response
        return state
    
    # For incomplete expenses, use the tool result directly
    if intent == "log_expense" and tool_result and tool_result.get("status") == "incomplete":
        state["final_response"] = tool_result.get("response", "I need more information to log your expense.")
        return state
    
    # For acknowledgments (thanks, okay, etc.), give a simple acknowledgment
    if intent == "acknowledgment":
        state["final_response"] = "You're welcome! Is there anything else I can help you with?"
        return state
    
    # For other cases, use LLM to generate natural response
    prompt = f"""
You are an intelligent expense tracking assistant. Generate a natural, helpful response based on the tool result.

User Message: "{user_message}"
Intent: {intent}
Tool Result: {json.dumps(tool_result, indent=2)}

Your task is to:
1. Take the tool result and convert it into a natural, conversational response
2. Be helpful and friendly
3. If there was an error, be apologetic and helpful
4. If clarification is needed, be clear about what's missing
5. If an expense was logged, confirm it clearly

RULES:
- Be conversational and natural
- Don't be robotic
- If expense was logged successfully, be positive and confirm
- If there's an error, be apologetic but helpful
- If clarification needed, be clear about what's missing
- Keep responses concise but informative

Generate ONLY the final response text, nothing else.
"""

    llm = ChatGroq(
        api_key=SecretStr(os.environ.get("GROQ_API_KEY") or ""),
        model=config.llm_model,
        temperature=0.7,  # Slightly higher for more natural responses
    )
    
    response = llm.invoke(prompt)
    final_response = response.content if hasattr(response, "content") else str(response)
    
    if isinstance(final_response, list):
        final_response = " ".join(str(x) for x in final_response)
    
    # Clean up the response
    final_response = final_response.strip()
    if final_response.startswith('"') and final_response.endswith('"'):
        final_response = final_response[1:-1]
    
    state["final_response"] = final_response
    logger.info(f"🤖 Final Response: {final_response}")
    
    return state

# --- Helper Functions ---
def parse_amount(amount_str):
    """Parse amount string like '750k' to 750000"""
    if not amount_str:
        return None
    
    amount_str = str(amount_str).lower().strip()
    
    # Handle common patterns
    if 'k' in amount_str:
        # Remove 'k' and multiply by 1000
        clean_amount = amount_str.replace('k', '').replace(',', '')
        try:
            return float(clean_amount) * 1000
        except ValueError:
            return None
    elif 'm' in amount_str:
        # Remove 'm' and multiply by 1000000
        clean_amount = amount_str.replace('m', '').replace(',', '')
        try:
            return float(clean_amount) * 1000000
        except ValueError:
            return None
    else:
        # Handle regular numbers with commas
        clean_amount = amount_str.replace(',', '')
        try:
            return float(clean_amount)
        except ValueError:
            return None

def map_category(item):
    """Map common items to categories"""
    if not item:
        return "misc"
    
    item_lower = str(item).lower()
    category_map = {
        # Food
        "breakfast": "food", "lunch": "food", "dinner": "food", "pizza": "food", 
        "burger": "food", "sandwich": "food", "coffee": "food", "tea": "food",
        "groceries": "food", "restaurant": "food", "meal": "food", "foodpanda": "food",
        "juice": "food", "milkshake": "food", "drinks": "food", "popcorn": "food",
        "snack": "food", "chips": "food", "candy": "food", "chocolate": "food",
        "ice cream": "food", "cake": "food", "bread": "food", "milk": "food",
        "eggs": "food", "meat": "food", "fish": "food", "vegetables": "food",
        "fruits": "food", "rice": "food", "pasta": "food", "soup": "food",
        "sweets": "food", "sweet": "food", "candy": "food", "chocolate bar": "food",
        "water": "food", "water bottles": "food", "water bottle": "food", "bottled water": "food",
        "apple": "food", "apples": "food", "carrot": "food", "carrots": "food", 
        "banana": "food", "bananas": "food", "orange": "food", "oranges": "food",
        "tomato": "food", "tomatoes": "food", "potato": "food", "potatoes": "food",
        "onion": "food", "onions": "food", "garlic": "food", "ginger": "food",
        
        # Transportation
        "car": "transportation", "bus": "transportation", "taxi": "transportation",
        "uber": "transportation", "fuel": "transportation", "gas": "transportation",
        "bus fare": "transportation", "fare": "transportation", "ticket": "transportation",
        "train": "transportation", "train ticket": "transportation", "metro": "transportation",
        "subway": "transportation", "bike": "transportation", "motorcycle": "transportation",
        
        # Electronics
        "phone": "electronics", "laptop": "electronics", "computer": "electronics",
        "charger": "electronics", "headphones": "electronics", "watch": "electronics",
        "tablet": "electronics", "camera": "electronics", "speaker": "electronics",
        "keyboard": "electronics", "mouse": "electronics", "gaming mouse": "electronics",
        "monitor": "electronics", "printer": "electronics", "scanner": "electronics",
        "webcam": "electronics", "microphone": "electronics", "router": "electronics",
        
        # Communication
        "phone balance": "communication", "balance": "communication", "calling": "communication",
        "mobile": "communication", "sim": "communication", "internet": "communication",
        "data": "communication", "sms": "communication", "call": "communication",
        
        # Stationery
        "notebook": "stationery", "pen": "stationery", "pencil": "stationery",
        "book": "stationery", "paper": "stationery", "folder": "stationery",
        "binder": "stationery", "stapler": "stationery", "scissors": "stationery",
        
        # Clothing
        "shirt": "clothing", "pants": "clothing", "shoes": "clothing",
        "dress": "clothing", "jacket": "clothing", "leather jacket": "clothing",
        "coat": "clothing", "sweater": "clothing", "jeans": "clothing",
        "hat": "clothing", "cap": "clothing", "scarf": "clothing", "gloves": "clothing",
        "socks": "clothing", "underwear": "clothing", "belt": "clothing",
        
        # Furniture/Home
        "chair": "furniture", "table": "furniture", "bed": "furniture",
        "sofa": "furniture", "desk": "furniture", "lamp": "furniture",
        "mirror": "furniture", "shelf": "furniture", "cabinet": "furniture",
        
        # Housing
        "rent": "housing", "apartment": "housing", "house": "housing",
        "electricity": "housing", "water": "housing", "gas": "housing",
        "maintenance": "housing", "repair": "housing",
        
        # Entertainment
        "movie": "entertainment", "cinema": "entertainment", "game": "entertainment",
        "concert": "entertainment", "ticket": "entertainment", "toy": "entertainment",
        "toy car": "entertainment", "video game": "entertainment", "music": "entertainment",
        "theater": "entertainment", "show": "entertainment", "amusement": "entertainment",
        "board game": "entertainment", "chess": "entertainment", "monopoly": "entertainment",
        "ludo": "entertainment", "scrabble": "entertainment", "puzzle": "entertainment",
        
        # Health
        "medicine": "health", "doctor": "health", "hospital": "health",
        "pharmacy": "health", "vitamins": "health", "dental": "health",
        "eye care": "health", "glasses": "health", "contact lenses": "health",
        
        # Sports
        "baseball": "sports", "baseball bat": "sports", "football": "sports",
        "basketball": "sports", "tennis": "sports", "gym": "sports",
        "fitness": "sports", "workout": "sports", "exercise": "sports",
        "cricket": "sports", "cricket kit": "sports", "cricket bat": "sports",
        "cricket ball": "sports", "cricket equipment": "sports", "sports kit": "sports",
        "badminton": "sports", "badminton racket": "sports", "badminton kit": "sports",
        "swimming": "sports", "swimming gear": "sports", "yoga": "sports",
        "yoga mat": "sports", "weights": "sports", "dumbbells": "sports",
        
        # Miscellaneous
        "keychain": "misc", "gift": "gift", "donation": "misc", "sent to": "gift", "transfer": "gift"
    }
    return category_map.get(item_lower, "misc")

# --- Graph Construction ---
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("llm_router", llm_router_node)
    workflow.add_node("log_expense_tool", log_expense_tool)
    workflow.add_node("query_expenses_tool", query_expenses_tool)
    workflow.add_node("get_total_expenses_tool", get_total_expenses_tool)
    workflow.add_node("greeting_tool", greeting_tool)
    workflow.add_node("clarification_tool", clarification_tool)
    workflow.add_node("log_multiple_expenses_tool", log_multiple_expenses_tool)
    workflow.add_node("final_response_node", final_response_node)
    
    # Routing from router to tools
    def route_to_tool(state: AgentState) -> str:
        tool_name = state.get("tool_name")
        if tool_name is None:
            return "clarification_tool"
        return tool_name
    
    workflow.add_conditional_edges(
        "llm_router",
        route_to_tool,
        {
            "log_expense_tool": "log_expense_tool",
            "query_expenses_tool": "query_expenses_tool",
            "get_total_expenses_tool": "get_total_expenses_tool",
            "greeting_tool": "greeting_tool",
            "clarification_tool": "clarification_tool",
            "log_multiple_expenses_tool": "log_multiple_expenses_tool"
        }
    )
    
    # All tools go to final response
    for tool in ["log_expense_tool", "query_expenses_tool", "get_total_expenses_tool", "greeting_tool", "clarification_tool", "log_multiple_expenses_tool"]:
        workflow.add_edge(tool, "final_response_node")
    
    # Final response ends the graph
    workflow.add_edge("final_response_node", END)
    workflow.set_entry_point("llm_router")
    
    return workflow.compile()

# --- Main Entry Point ---
def process_message_with_agent(phone_number: str, message: str, db) -> Optional[Dict[str, Any]]:
    if not config.enabled:
        return None
    try:
        logger.info(f"🤖 INTELLIGENT AGENT: Processing message: {message}")
        conversation_history = memory.get_conversation_context(phone_number)
        pending_expense = memory.get_pending_expense(phone_number)
        
        from ..models import User
        user = db.query(User).filter(User.phone_number == phone_number).first()
        if not user:
            return None
        user_id = getattr(user, "id")
        
        state = AgentState(
            phone_number=phone_number,
            user_message=message,
            conversation_history=[{
                "user": turn.user_message,
                "assistant": turn.bot_response
            } for turn in conversation_history[-3:]],
            user_id=user_id,
            response=None,
            pending_expense=pending_expense,
            intent=None,
            amount=None,
            category=None,
            item=None,
            clarification=None,
            db=db,
            tool_result=None,
            tool_name=None,
            final_response=None,
            multiple_expenses=None
        )
        
        graph = create_agent_graph()
        final_state = graph.invoke(state)
        
        final_response = final_state.get("final_response")
        if final_response:
            memory.add_conversation_turn(
                phone_number=phone_number,
                user_message=message,
                bot_response=final_response,
                intent=final_state.get("intent") or "intelligent_agent",
                confidence=0.9
            )
            logger.info(f"🤖 INTELLIGENT AGENT: Success! Returning: {final_response}")
            return {
                "message": final_response,
                "intent": final_state.get("intent") or "intelligent_agent",
                "confidence": 0.9,
                "tools_used": [final_state.get("tool_name")]
            }
        
        logger.info(f"🤖 INTELLIGENT AGENT: No response generated")
        return None
        
    except Exception as e:
        logger.error(f"Agent processing error: {e}")
        import traceback
        traceback.print_exc()
        return None 