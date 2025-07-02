# tools.py

#from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from typing import Dict, Any
from groq import Groq
import json, re
from dotenv import load_dotenv
from app import crud, models
from app.intelligent_agent_v3.config import config
from datetime import datetime, timedelta

load_dotenv()

# Set up Groq client
llm_client = Groq(api_key=config.groq_api_key)


def clean_json_response(raw_response: str) -> str:
    """Clean LLM response to extract valid JSON"""
    if not raw_response:
        return "{}"
    
    # Remove markdown code blocks
    raw_response = re.sub(r'```(?:json)?\s*', '', raw_response)
    raw_response = re.sub(r'```\s*', '', raw_response)
    
    # Convert single quotes to double quotes
    raw_response = raw_response.replace("'", '"')
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return raw_response


# Add context management functions
def get_conversation_context(db, phone_number: str):
    """Retrieve stored conversation context"""
    try:
        user = crud.get_user_by_phone_number(db, phone_number)
        if not user:
            return {}
        
        # For now, let's use a simple approach - store in a new table or 
        # use a cache. For immediate fix, let's add a context field to User model
        # But first, let's implement a simpler solution using a global cache
        
        # Check if context exists and is recent (within 5 minutes)
        cache_key = f"context_{phone_number}"
        if hasattr(get_conversation_context, '_cache'):
            cached = get_conversation_context._cache.get(cache_key)
            if cached and cached.get('timestamp'):
                if datetime.now() - cached['timestamp'] < timedelta(minutes=5):
                    return cached.get('context', {})
        
        return {}
    except:
        return {}

def store_conversation_context(db, phone_number: str, context: dict):
    """Store conversation context - only keep the most recent incomplete expense"""
    try:
        if not hasattr(store_conversation_context, '_cache'):
            store_conversation_context._cache = {}
        
        cache_key = f"context_{phone_number}"
        
        # If context is empty or None, clear the cache
        if not context:
            if cache_key in store_conversation_context._cache:
                del store_conversation_context._cache[cache_key]
            return
        
        # Only store single context dict, not lists
        if isinstance(context, dict):
            store_conversation_context._cache[cache_key] = {
                'context': context,
                'timestamp': datetime.now()
            }
        else:
            # If somehow a list is passed, take the last item
            if isinstance(context, list) and context:
                store_conversation_context._cache[cache_key] = {
                    'context': context[-1],  # Take most recent
                    'timestamp': datetime.now()
                }
        
        # Also store reference in get_conversation_context
        get_conversation_context._cache = store_conversation_context._cache
    except Exception as e:
        print(f"[ERROR] Failed to store context: {e}")


# 1. Intent Detection Tool
class IntentTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] IntentTool invoked with state:", state)
        message = state["message"]
        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": (
                    "You are a WhatsApp expense bot. Classify the user's intent as one of: "
                    "'log_expense', 'query', 'breakdown', 'chitchat'. "
                    "\n\nGuidelines:"
                    "\n- 'query': Specific questions like 'top 5 expenses', 'most expensive', 'cheapest', 'how much did I spend on X', 'compare months', 'breakdown for past week/month', 'expenses in January', etc."
                    "\n- 'breakdown': General requests like 'spending breakdown', 'category summary', 'overall breakdown' (without time periods)"
                    "\n- 'log_expense': Adding/recording new expenses"
                    "\n- 'chitchat': General conversation, greetings, non-expense related"
                    "\nReturn ONLY a valid JSON object with double quotes. "
                    "Example: {\"intent\": \"query\"}"
                )},
                {"role": "user", "content": message},
            ]
        )
        raw = response.choices[0].message.content
        if raw is None:
            print("[WARNING] IntentTool: LLM returned None response.")
            return {**state, "intent": "chitchat"}
        
        try:
            cleaned_json = clean_json_response(raw)
            result = json.loads(cleaned_json) if cleaned_json else {}
            intent = result.get("intent", "chitchat")
            print("[DEBUG] IntentTool output:", result)
            return {**state, "intent": intent}
        except Exception as e:
            print(f"[ERROR] IntentTool: Failed to parse LLM response: {e}")
            return {**state, "intent": "chitchat"}


# 2. Completely Rewritten Expense Extraction Tool
class ExtractExpenseTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] ExtractExpenseTool invoked with state:", state)
        db: Any = state.get("db")
        message = state["message"]
        phone_number = state["phone_number"]

        # Get or create user
        db_user = crud.get_user_by_phone_number(db, phone_number) if db else None
        if db and not db_user:
            db_user = crud.create_user(db, user=models.User(phone_number=phone_number))

        # Check for pending context from previous messages
        pending_context = state.get("pending_context", {})
        
        # Fix: Handle case where pending_context is a list
        if isinstance(pending_context, list):
            pending_context = pending_context[-1] if pending_context else {}
            print(f"[DEBUG] Fixed list context to: {pending_context}")
        
        # Enhanced context processing
        if pending_context:
            enhanced_message = self._enhance_message_with_context(message, pending_context)
            print(f"[DEBUG] Enhanced message with context: {enhanced_message}")
        else:
            enhanced_message = message

        # Use LLM to extract expenses intelligently
        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": (
                    "You are an intelligent expense extraction system. Extract expenses from user messages.\n\n"
                    "IMPORTANT RULES:\n"
                    "1. Extract ALL complete expenses (both amount and item mentioned)\n"
                    "2. Handle multiple expenses in one message: 'soccer ball: 8k, shoes: 11k, socks: 800'\n"
                    "3. Handle various formats: '8000', '8k', '8K', '8,000'\n"
                    "4. Smart categorization: food/groceries, transportation/transport, entertainment, shopping/clothing, health, electronics, sports, rent/housing, other\n"
                    "5. Extract meaningful notes from item descriptions\n"
                    "6. If message has amount AND item, it's COMPLETE - extract it\n"
                    "7. If missing amount OR item, mark as incomplete\n\n"
                    "Return JSON format:\n"
                    "{\n"
                    '  "complete_expenses": [{"amount": 8000, "category": "sports", "note": "soccer ball"}],\n'
                    '  "incomplete_expense": {"type": "missing_amount", "item": "shoes"} OR {"type": "missing_item", "amount": 5000} OR null\n'
                    "}\n\n"
                    "Examples:\n"
                    "- 'soccer ball 8k' → complete\n"
                    "- 'I spent 500 PKR' → incomplete (missing item)\n"
                    "- 'bought shoes' → incomplete (missing amount)\n"
                    "- 'phone 25k, lunch 300' → both complete"
                )},
                {"role": "user", "content": enhanced_message},
            ]
        )
        
        raw = response.choices[0].message.content
        print(f"[DEBUG] ExtractExpenseTool raw LLM output: {raw}")
        
        try:
            if raw is None:
                raw = "{}"
            cleaned_json = clean_json_response(raw)
            result = json.loads(cleaned_json) if cleaned_json else {}
            
            complete_expenses = result.get("complete_expenses", [])
            incomplete_expense = result.get("incomplete_expense")
            
            print(f"[DEBUG] ExtractExpenseTool extracted: {len(complete_expenses)} complete, incomplete: {incomplete_expense}")
            
            # Prepare return state
            new_state = {**state, "db_user": db_user, "expenses": complete_expenses}
            
            # Handle incomplete expense - ensure it's a dict, not a list
            if incomplete_expense and isinstance(incomplete_expense, dict):
                new_state["pending_context"] = incomplete_expense
            else:
                new_state["pending_context"] = {}
                
            return new_state
            
        except Exception as e:
            print(f"[ERROR] ExtractExpenseTool: Failed to parse LLM response: {e}")
            return {**state, "db_user": db_user, "expenses": [], "pending_context": {}}

    def _enhance_message_with_context(self, message: str, pending_context: dict) -> str:
        """Enhance current message with pending context intelligently"""
        if pending_context.get("type") == "missing_amount" and pending_context.get("item"):
            # We were asking for amount, now they provided it
            item = pending_context["item"]
            return f"{message} for {item}"
        elif pending_context.get("type") == "missing_item" and pending_context.get("amount"):
            # We were asking for item, now they provided it
            amount = pending_context["amount"]
            return f"{amount} PKR for {message}"
        return message


# 3. Completely Rewritten Expense Creation Tool
class CreateExpenseTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] CreateExpenseTool invoked with state:", state)
        db: Any = state.get("db")
        user = state["db_user"]
        user_id = user.id if user else None
        expenses = state.get("expenses", [])
        pending_context = state.get("pending_context", {})
        message = state["message"]
        phone_number = state["phone_number"]

        # Fix: Handle case where pending_context is a list
        if isinstance(pending_context, list):
            pending_context = pending_context[-1] if pending_context else {}
            print(f"[DEBUG] CreateExpenseTool: Fixed list context to: {pending_context}")

        # If we have complete expenses, log them
        if expenses and db and user_id:
            inserted = []
            for expense in expenses:
                try:
                    amount = expense.get("amount")
                    category_name = expense.get("category")
                    note = expense.get("note", "")

                    if amount and category_name:
                        category = crud.get_or_create_category(db, user_id, category_name)
                        category_id = getattr(category, "id", None)
                        if category_id is not None:
                            new_expense = crud.create_expense(db, user_id, category_id, amount, note)
                            inserted.append(new_expense)
                except Exception as e:
                    print(f"[ERROR] CreateExpenseTool: Failed to create expense: {e}")

            if inserted:
                # Clear context after successful expense logging
                store_conversation_context(db, phone_number, {})
                response_message = self._generate_intelligent_success_message(inserted, db)
                return {**state, "final_response": response_message, "pending_context": {}}

        # If we have pending context, store it and generate clarification
        if pending_context:
            store_conversation_context(db, phone_number, pending_context)
            clarification = self._generate_intelligent_clarification(pending_context, message)
            return {**state, "final_response": clarification}

        # If no expenses extracted and no pending context
        if not expenses and not pending_context:
            # Clear any stale context
            store_conversation_context(db, phone_number, {})
            no_expense_response = self._generate_no_expense_response(message)
            return {**state, "final_response": no_expense_response, "pending_context": {}}

        return {**state, "final_response": "I'm having trouble understanding that expense. Could you try rephrasing it?"}

    def _generate_intelligent_success_message(self, inserted_expenses, db):
        """Generate natural success message using LLM"""
        if len(inserted_expenses) == 1:
            expense = inserted_expenses[0]
            category = db.query(models.Category).filter(models.Category.id == expense.category_id).first()
            category_name = category.name if category else "unknown"
            note_text = f" ({expense.note})" if expense.note else ""
            
            prompt = f"The user successfully logged an expense: {expense.amount} PKR for {category_name}{note_text}. Generate a natural, encouraging confirmation message. Be conversational and friendly."
        else:
            total = sum(exp.amount for exp in inserted_expenses)
            prompt = f"The user successfully logged {len(inserted_expenses)} expenses totaling {total} PKR. Generate a natural, encouraging confirmation message that mentions the count and total. Be conversational and friendly."

        try:
            response = llm_client.chat.completions.create(
                model=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant. Generate brief, natural confirmation messages for logged expenses. Be encouraging and conversational."},
                    {"role": "user", "content": prompt},
                ]
            )
            
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception as e:
            print(f"[ERROR] Failed to generate success message: {e}")

        # Fallback
        if len(inserted_expenses) == 1:
            expense = inserted_expenses[0]
            category = db.query(models.Category).filter(models.Category.id == expense.category_id).first()
            category_name = category.name if category else "unknown"
            return f"✅ Got it! Logged {expense.amount} PKR for {category_name}."
        else:
            total = sum(exp.amount for exp in inserted_expenses)
            return f"✅ Perfect! Logged {len(inserted_expenses)} expenses totaling {total} PKR."

    def _generate_intelligent_clarification(self, pending_context, original_message):
        """Generate natural clarification question using LLM"""
        if pending_context.get("type") == "missing_amount":
            item = pending_context.get("item", "that item")
            prompt = f"The user mentioned buying '{item}' but didn't say how much it cost. Generate a natural, conversational question asking for the amount in PKR. Be friendly and specific about the item."
        elif pending_context.get("type") == "missing_item":
            amount = pending_context.get("amount", "some money")
            prompt = f"The user said they spent {amount} PKR but didn't mention what they bought. Generate a natural, conversational question asking what they purchased. Be friendly."
        else:
            prompt = f"The user said '{original_message}' but it's unclear what expense they want to log. Generate a natural, helpful question to clarify."

        try:
            response = llm_client.chat.completions.create(
                model=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant. Generate natural, conversational questions to clarify incomplete expense information. Be friendly and specific."},
                    {"role": "user", "content": prompt},
                ]
            )
            
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception as e:
            print(f"[ERROR] Failed to generate clarification: {e}")

        # Fallback
        if pending_context.get("type") == "missing_amount":
            item = pending_context.get("item", "that")
            return f"How much did you spend on {item}? (in PKR)"
        elif pending_context.get("type") == "missing_item":
            amount = pending_context.get("amount", "that amount")
            return f"What did you spend {amount} PKR on?"
        else:
            return "Could you tell me what you bought and how much you spent? (in PKR)"

    def _generate_no_expense_response(self, message):
        """Generate intelligent response when no expense is detected"""
        prompt = f"The user said '{message}' but I couldn't detect any specific expense to log. Generate a natural, helpful response that encourages them to share expense details in PKR. Be conversational and give an example."

        try:
            response = llm_client.chat.completions.create(
                model=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant. When users mention expenses but don't provide enough detail, guide them naturally. Be encouraging and give examples."},
                    {"role": "user", "content": prompt},
                ]
            )
            
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception as e:
            print(f"[ERROR] Failed to generate no expense response: {e}")

        # Fallback
        return "I'd love to help you log that expense! Could you tell me what you bought and how much you spent? For example: 'I spent 500 PKR on lunch' or 'bought groceries for 2000 PKR'."


# 4. SQL Generation Tool (for queries)
class GenerateSQLTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] GenerateSQLTool invoked with state:", state)
        print("[DEBUG] GenerateSQLTool run_config db:", run_config.get('db'))
        # Get user from database if not already in state
        db: Any = state.get("db")
        phone_number = state["phone_number"]
        db_user = crud.get_user_by_phone_number(db, phone_number) if db else None
        if db and not db_user:
            db_user = crud.create_user(db, user=models.User(phone_number=phone_number))
        
        user_id = db_user.id if db_user else None
        message = state["message"]

        sql_prompt = (
            f"You are a PostgreSQL expert helping generate SQL for an expense tracker.\n"
            f"User data is stored in tables: users(id), categories(id, name, user_id), expenses(id, user_id, category_id, amount, timestamp, note).\n"
            f"Only generate SELECT statements to answer the user's question.\n"
            f"Always filter with e.user_id = {user_id}.\n"
            f"Respond with only the SQL query, no markdown formatting or code blocks."
        )

        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": sql_prompt},
                {"role": "user", "content": message},
            ]
        )
        sql = response.choices[0].message.content
        # Clean SQL by removing markdown code blocks
        if sql:
            sql = sql.strip()
            # Remove ```sql and ``` markers
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()
        else:
            sql = ""
        print("[DEBUG] GenerateSQLTool output SQL:", sql)
        return {**state, "sql": sql, "db_user": db_user}


# 5. SQL Execution Tool
class ExecuteSQLTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] ExecuteSQLTool invoked with state:", state)
        db: Any = state.get("db")
        sql = state.get("sql")
        
        # Clean SQL by removing any remaining markdown formatting
        if sql:
            sql = sql.strip()
            # Remove ```sql and ``` markers if still present
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()
        
        if not sql or not db:
            print("[WARNING] ExecuteSQLTool: No SQL or DB provided.")
            return {**state, "sql_result": None}

        try:
            from sqlalchemy import text as sql_text
            result = db.execute(sql_text(sql))
            rows = result.fetchall()
            
            # Always return the raw result, even if None or empty
            if not rows:
                formatted = None
            elif len(rows) == 1 and len(rows[0]) == 1:
                formatted = rows[0][0]
            else:
                formatted = rows
                
            print("[DEBUG] ExecuteSQLTool output:", formatted)
            return {**state, "sql_result": formatted}
        except Exception as e:
            print(f"[ERROR] ExecuteSQLTool: SQL execution error: {e}")
            # Return None so FormatQueryResponseTool can handle it gracefully
            return {**state, "sql_result": None}


# 6. Breakdown Formatter Tool
class FormatBreakdownTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] FormatBreakdownTool invoked with state:", state)
        
        # Get user from database if not already in state
        db: Any = state.get("db")
        phone_number = state["phone_number"]
        
        try:
            db_user = crud.get_user_by_phone_number(db, phone_number) if db else None
            if db and not db_user:
                db_user = crud.create_user(db, user=models.User(phone_number=phone_number))
            
            user_id = db_user.id if db_user else None
            
            # Fix the linter error by being more explicit
            if user_id is None:
                return {**state, "final_response": "I couldn't find your account. Please try logging an expense first."}
            
            query = """
            SELECT c.name, SUM(e.amount), COUNT(*)
            FROM expenses e
            JOIN categories c ON e.category_id = c.id
            WHERE e.user_id = :user_id
            GROUP BY c.name
            ORDER BY SUM(e.amount) DESC
            """
            
            if not db:
                return {**state, "final_response": "I'm having trouble accessing your data right now. Please try again in a moment."}
            
            from sqlalchemy import text as sql_text
            result = db.execute(sql_text(query), {"user_id": user_id}).fetchall()
            
            if not result:
                return {**state, "final_response": "You haven't logged any expenses yet! Start by telling me about a purchase you made."}

            lines = ["📊 Your Spending Breakdown:"]
            total = 0
            for cat, amount, count in result:
                total += amount
                lines.append(f"• {cat.title()}: PKR {amount:,.0f} ({count} items)")
            lines.append(f"\nTotal Spent: PKR {total:,.0f}")

            output = "\n".join(lines)
            print("[DEBUG] FormatBreakdownTool output:", output)
            return {**state, "final_response": output}
            
        except Exception as e:
            print(f"[ERROR] FormatBreakdownTool: {e}")
            return {**state, "final_response": "I'm having trouble getting your spending breakdown right now. Please try again!"}


# 7. Chitchat Tool (Fallback for unknown intent)
class ChitchatTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] ChitchatTool invoked with state:", state)
        print("[DEBUG] ChitchatTool run_config db:", run_config.get('db'))
        message = state["message"]
        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": (
                    "You are a friendly financial assistant for expense tracking. "
                    "Reply casually but helpfully to any general messages. "
                    "Focus on expense tracking capabilities and be encouraging about financial management."
                )},
                {"role": "user", "content": message},
            ]
        )
        reply = response.choices[0].message.content
        if reply is None or not reply.strip():
            print("[WARNING] ChitchatTool: LLM returned empty response. Using fallback.")
            reply = "Hello! How can I help you with your expenses today?"
        else:
            reply = reply.strip()
        print("[DEBUG] ChitchatTool output:", reply)
        return {**state, "final_response": reply}


# 8. Respond Tool (Pass-through, for fallback/default end state)
class RespondTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] RespondTool invoked with state:", state)
        print("[DEBUG] RespondTool run_config db:", run_config.get('db'))
        # Only pass through final_response or sql_result, do not generate fallback/template messages
        if state.get("final_response"):
            return state
        elif state.get("sql_result") is not None:
            return {**state, "final_response": str(state["sql_result"])}
        else:
            return state


# Fix Query Response Tool to always use LLM
class FormatQueryResponseTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] FormatQueryResponseTool invoked with state:", state)
        
        message = state["message"]
        sql_result = state.get("sql_result")
        
        try:
            # Always use LLM to format response, even for empty results
            if sql_result is None:
                format_prompt = f"""
The user asked: "{message}"

The database search didn't return any results - this means there are no expenses matching what they're looking for.

Generate a natural, conversational response that:
- Doesn't use technical terms like "data", "query", "criteria", "records"
- Sounds like a helpful friend talking about expenses
- Is encouraging and suggests next steps
- Is specific to what they asked about

Be natural and conversational, not robotic.
"""
            else:
                format_prompt = f"""
The user asked: "{message}"

The database returned this result: {sql_result}

Format this into a natural, friendly response that:
- Uses proper currency formatting (PKR)
- Is conversational and user-friendly
- Provides context and explanation
- Uses appropriate formatting (commas for large numbers)

Respond as if you're talking directly to the user about their expenses.
"""
            
            response = llm_client.chat.completions.create(
                model=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant that provides clear, friendly, conversational responses. Never use technical terms like 'data', 'query', 'criteria', 'records'. Talk like a friendly helper."},
                    {"role": "user", "content": format_prompt},
                ]
            )
            
            formatted_response = response.choices[0].message.content
            if formatted_response is None or not formatted_response.strip():
                # This should rarely happen, but just in case
                formatted_response = "I'm having trouble understanding that right now. Could you try asking in a different way?"
            else:
                formatted_response = formatted_response.strip()
            
            print("[DEBUG] FormatQueryResponseTool output:", formatted_response)
            return {**state, "final_response": formatted_response}
            
        except Exception as e:
            print(f"[ERROR] FormatQueryResponseTool: {e}")
            # Even error handling should be LLM-generated, but as a last resort fallback
            return {**state, "final_response": "I'm having trouble with that right now. Could you try asking me something else about your expenses?"}