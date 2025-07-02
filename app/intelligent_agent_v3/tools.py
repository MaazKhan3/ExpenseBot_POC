# tools.py

#from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from typing import Dict, Any, Optional
from groq import Groq
import json, re
from dotenv import load_dotenv
from app import crud, models
from app.intelligent_agent_v3.config import config

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


# 1. Intent Detection Tool
class IntentTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] IntentTool invoked with state:", state)
        print("[DEBUG] IntentTool run_config db:", run_config.get('db'))
        message = state["message"]
        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": (
                    "You are a WhatsApp expense bot. Classify the user's intent as one of: "
                    "'log_expense', 'query', 'breakdown', 'chitchat'. "
                    "Return ONLY a valid JSON object with double quotes. "
                    "Example: {\"intent\": \"log_expense\"}"
                )},
                {"role": "user", "content": message},
            ]
        )
        raw = response.choices[0].message.content
        if raw is None:
            print("[WARNING] IntentTool: LLM returned None response.")
            return {**state, "intent": "unknown"}
        
        try:
            cleaned_json = clean_json_response(raw)
            result = json.loads(cleaned_json) if cleaned_json else {}
            print("[DEBUG] IntentTool output:", result)
            return {**state, "intent": result.get("intent", "unknown")}
        except Exception as e:
            print(f"[ERROR] IntentTool: Failed to parse LLM response: {e}")
            print(f"[ERROR] Raw response: {raw}")
            return {**state, "intent": "unknown"}


# 2. Expense Extraction Tool
class ExtractExpenseTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] ExtractExpenseTool invoked with state:", state)
        print("[DEBUG] ExtractExpenseTool run_config db:", run_config.get('db'))
        db: Any = state.get("db")  # Get db from state instead of run_config
        message = state["message"]
        phone_number = state["phone_number"]

        db_user = crud.get_user_by_phone_number(db, phone_number) if db else None
        if db and not db_user:
            db_user = crud.create_user(db, user=models.User(phone_number=phone_number))

        # Check for pending context from previous messages
        pending_context = state.get("pending_context", {})
        
        # If we have pending context, try to merge it with current message
        if pending_context:
            enhanced_message = self._enhance_message_with_context(message, pending_context)
            print(f"[DEBUG] Enhanced message with context: {enhanced_message}")
        else:
            enhanced_message = message

        # First, check if this is an incomplete expense
        is_incomplete = self._detect_incomplete_expense(enhanced_message)
        
        if is_incomplete:
            # Handle incomplete expense
            return self._handle_incomplete_expense(enhanced_message, state, db_user)
        
        response = llm_client.chat.completions.create(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": (
                    "Extract expenses from user message. Return ONLY valid JSON with double quotes. "
                    "Format: {\"expenses\": [{\"amount\": 500, \"category\": \"groceries\", \"note\": \"optional\"}]} "
                    "Use smart categorization: food (groceries, lunch, dinner, snacks), transportation (fuel, taxi, bus), "
                    "entertainment (movies, games, sports), shopping (clothes, electronics), health (medicine, doctor), "
                    "education (books, courses), utilities (electricity, water), sports (equipment, gym, tennis, cricket), other. "
                    "Always extract meaningful notes from the item description. "
                    "For sports equipment like tennis racket, categorize as 'sports'. "
                    "For food items like apples, mangoes, categorize as 'food'."
                )},
                {"role": "user", "content": enhanced_message},
            ]
        )
        raw = response.choices[0].message.content
        print("[DEBUG] ExtractExpenseTool raw LLM output:", raw)
        if raw is None:
            print("[WARNING] ExtractExpenseTool: LLM returned None response.")
            return {**state, "db_user": db_user, "expenses": []}
        
        try:
            cleaned_json = clean_json_response(raw)
            data = json.loads(cleaned_json) if cleaned_json else {}
            expenses = data.get("expenses", [])
            print("[DEBUG] ExtractExpenseTool output:", expenses)
            return {**state, "db_user": db_user, "expenses": expenses}
        except Exception as e:
            print(f"[ERROR] ExtractExpenseTool: Failed to parse LLM response: {e}")
            print(f"[ERROR] Raw response: {raw}")
            return {**state, "db_user": db_user, "expenses": []}
    
    def _detect_incomplete_expense(self, message: str) -> bool:
        """Detect if the message represents an incomplete expense"""
        message_lower = message.lower()
        
        # Check for patterns that indicate incomplete expenses
        incomplete_patterns = [
            # Missing amount patterns
            "bought", "got", "purchased", "bought a", "got a", "purchased a",
            "tennis racket", "shoes", "book", "phone", "laptop",
            # Missing item patterns  
            "spent", "paid", "cost", "expense",
            # Just numbers
            lambda msg: msg.strip().replace(',', '').replace('k', '').replace('m', '').isdigit()
        ]
        
        for pattern in incomplete_patterns:
            if callable(pattern):
                if pattern(message_lower):
                    return True
            elif pattern in message_lower:
                return True
        
        return False
    
    def _handle_incomplete_expense(self, message: str, state: Dict[str, Any], db_user) -> Dict[str, Any]:
        """Handle incomplete expense by setting up pending context"""
        message_lower = message.lower()
        
        # Check if it's missing amount (has item but no amount)
        if any(word in message_lower for word in ["bought", "got", "purchased", "tennis racket", "shoes"]):
            # Extract the item name
            item_name = self._extract_item_name(message)
            pending_context = {"item": item_name, "type": "missing_amount"}
            return {**state, "db_user": db_user, "expenses": [], "pending_context": pending_context}
        
        # Check if it's missing item (has amount but no item)
        elif any(word in message_lower for word in ["spent", "paid", "cost"]):
            # Extract the amount
            amount = self._extract_amount(message)
            if amount:
                pending_context = {"amount": amount, "type": "missing_item"}
                return {**state, "db_user": db_user, "expenses": [], "pending_context": pending_context}
        
        # Just a number
        elif message.strip().replace(',', '').replace('k', '').replace('m', '').isdigit():
            amount = self._extract_amount(message)
            if amount:
                pending_context = {"amount": amount, "type": "missing_item"}
                return {**state, "db_user": db_user, "expenses": [], "pending_context": pending_context}
        
        return {**state, "db_user": db_user, "expenses": []}
    
    def _extract_item_name(self, message: str) -> str:
        """Extract item name from incomplete expense message"""
        # Simple extraction - look for common patterns
        if "tennis racket" in message.lower():
            return "tennis racket"
        elif "shoes" in message.lower():
            return "shoes"
        elif "bought" in message.lower():
            # Extract what comes after "bought"
            parts = message.lower().split("bought")
            if len(parts) > 1:
                item = parts[1].strip()
                # Remove common words
                item = item.replace("a ", "").replace("an ", "").replace("the ", "")
                return item
        return "item"
    
    def _extract_amount(self, message: str) -> Optional[float]:
        """Extract amount from message"""
        import re
        
        # Look for numbers with k, m suffixes
        patterns = [
            r'(\d+(?:\.\d+)?)k',  # 2k, 1.5k
            r'(\d+(?:\.\d+)?)m',  # 1m, 1.5m
            r'(\d+(?:,\d+)*)',    # 1,000, 2,500
            r'(\d+(?:\.\d+)?)',   # 500, 1.5
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                value = match.group(1)
                if 'k' in message[match.start():match.end()]:
                    return float(value) * 1000
                elif 'm' in message[match.start():match.end()]:
                    return float(value) * 1000000
                else:
                    return float(value.replace(',', ''))
        
        return None
    
    def _enhance_message_with_context(self, message: str, pending_context: dict) -> str:
        """Enhance current message with pending context"""
        if pending_context.get("item") and not any(word in message.lower() for word in ["spent", "bought", "cost", "paid"]):
            # If we have a pending item and current message looks like just an amount
            return f"{message} for {pending_context['item']}"
        elif pending_context.get("amount") and not any(char.isdigit() for char in message):
            # If we have a pending amount and current message looks like just an item
            return f"{pending_context['amount']} for {message}"
        return message


# 3. Expense Creation Tool
class CreateExpenseTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] CreateExpenseTool invoked with state:", state)
        print("[DEBUG] CreateExpenseTool run_config db:", run_config.get('db'))
        db: Any = state.get("db")  # Get db from state instead of run_config
        user = state["db_user"]
        user_id = user.id if user else None
        expenses = state.get("expenses", [])

        inserted = []
        incomplete_expenses = []
        new_pending_context = {}
        
        if db and user_id:
            for item in expenses:
                amount = item.get("amount")
                category_name = item.get("category")
                note = item.get("note", "")

                if not amount or not category_name:
                    incomplete_expenses.append(item)
                    # Store pending context for next message
                    if not amount and category_name:
                        new_pending_context = {"item": category_name, "type": "missing_amount"}
                    elif amount and not category_name:
                        new_pending_context = {"amount": amount, "type": "missing_item"}
                    continue
                    
                category = crud.get_or_create_category(db, user_id, category_name)
                category_id = getattr(category, "id", None)
                if category_id is not None:
                    expense = crud.create_expense(db, user_id, category_id, amount, note)
                    inserted.append(expense)

        if incomplete_expenses:
            # Generate clarification message for incomplete expenses
            clarification_msg = self._generate_clarification_message(incomplete_expenses)
            msg = clarification_msg
            # Store pending context for next message
            state["pending_context"] = new_pending_context
        elif state.get("pending_context"):
            # We have pending context but no expenses to process
            clarification_msg = self._generate_clarification_message_from_context(state["pending_context"])
            msg = clarification_msg
        else:
            # Generate detailed success message
            msg = self._generate_detailed_success_message(inserted, db)
            # Clear any pending context since we successfully logged everything
            state["pending_context"] = {}
            
        print("[DEBUG] CreateExpenseTool output:", msg)
        return {**state, "final_response": msg}
    
    def _generate_clarification_message(self, incomplete_expenses):
        """Generate a natural clarification message for incomplete expenses"""
        if len(incomplete_expenses) == 1:
            expense = incomplete_expenses[0]
            if not expense.get("amount") and expense.get("category"):
                item_name = expense['category']
                return f"How much did you spend on {item_name}?"
            elif expense.get("amount") and not expense.get("category"):
                amount = expense['amount']
                return f"What did you spend {amount} PKR on?"
        else:
            return "I need more details for some expenses. Could you provide the missing amounts or categories?"
        
        return "I need more information to log this expense properly. Could you provide the missing details?"
    
    def _generate_clarification_message_from_context(self, pending_context):
        """Generate clarification message from pending context"""
        if pending_context.get("type") == "missing_amount":
            item = pending_context.get("item", "this item")
            return f"How much did you spend on {item}?"
        elif pending_context.get("type") == "missing_item":
            amount = pending_context.get("amount")
            return f"What did you spend {amount} PKR on?"
        else:
            return "I need more information to log this expense properly. Could you provide the missing details?"
    
    def _generate_detailed_success_message(self, inserted_expenses, db):
        """Generate a detailed success message showing what was logged"""
        if not inserted_expenses:
            return "No expenses were logged."
        
        if len(inserted_expenses) == 1:
            expense = inserted_expenses[0]
            category = db.query(models.Category).filter(models.Category.id == expense.category_id).first()
            category_name = category.name if category else "unknown"
            note_text = f" ({expense.note})" if expense.note else ""
            return f"✅ Logged {expense.amount} PKR for {category_name}{note_text}"
        else:
            lines = [f"✅ Logged {len(inserted_expenses)} expenses:"]
            total = 0
            for expense in inserted_expenses:
                category = db.query(models.Category).filter(models.Category.id == expense.category_id).first()
                category_name = category.name if category else "unknown"
                note_text = f" ({expense.note})" if expense.note else ""
                lines.append(f"• {expense.amount} PKR - {category_name}{note_text}")
                total += expense.amount
            lines.append(f"Total: {total} PKR")
            return "\n".join(lines)


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
        print("[DEBUG] ExecuteSQLTool run_config db:", run_config.get('db'))
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
            colnames = result.keys()
            # Always return the raw result, even if None or empty
            if len(rows) == 1 and len(rows[0]) == 1:
                value = rows[0][0]
                formatted = value
            else:
                formatted = rows
            print("[DEBUG] ExecuteSQLTool output:", formatted)
            return {**state, "sql_result": formatted}
        except Exception as e:
            print(f"[ERROR] ExecuteSQLTool: SQL execution error: {e}")
            return {**state, "sql_result": f"SQL Error: {e}"}


# 6. Breakdown Formatter Tool
class FormatBreakdownTool(Runnable):
    def invoke(self, state: Dict[str, Any], run_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        print("[DEBUG] FormatBreakdownTool invoked with state:", state)
        print("[DEBUG] FormatBreakdownTool run_config db:", run_config.get('db'))
        user_id = state["db_user"].id if state.get("db_user") else None
        query = f"""
        SELECT c.name, SUM(e.amount), COUNT(*)
        FROM expenses e
        JOIN categories c ON e.category_id = c.id
        WHERE e.user_id = {user_id}
        GROUP BY c.name
        ORDER BY SUM(e.amount) DESC
        """
        db: Any = state.get("db")  # Get db from state instead of run_config
        
        if not db:
            print("[DEBUG] FormatBreakdownTool: No database connection.")
            return {**state, "final_response": "Database connection not available."}
        
        try:
            from sqlalchemy import text as sql_text
            result = db.execute(sql_text(query)).fetchall()
        except Exception as e:
            print(f"[ERROR] FormatBreakdownTool: SQL execution error: {e}")
            return {**state, "final_response": f"Error retrieving breakdown: {e}"}

        if not result:
            print("[DEBUG] FormatBreakdownTool: No expenses found.")
            return {**state, "final_response": "No expenses found."}

        lines = [f"📊 Spending Breakdown:"]
        total = 0
        for cat, amount, count in result:
            total += amount
            lines.append(f"• {cat.title()}: PKR {amount:,.0f} ({count} items)")
        lines.append(f"Total: PKR {total:,.0f}")

        output = "\n".join(lines)
        print("[DEBUG] FormatBreakdownTool output:", output)
        return {**state, "final_response": output}


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