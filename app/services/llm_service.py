import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

SYSTEM_PROMPT = (
    "You are an intelligent expense tracker assistant. "
    "Given a user's WhatsApp message, classify the intent as one of: 'expense_logging', 'query', 'breakdown', or 'management'. "
    "If intent is 'expense_logging', extract all (amount, category, note) tuples from the message. "
    "If intent is 'query', extract the query type and any relevant time ranges or categories. "
    "If intent is 'breakdown', this means the user wants to see a breakdown/categorization of their expenses. "
    "Respond ONLY in the following JSON format:\n"
    "{\n"
    "  'intent': 'expense_logging' | 'query' | 'breakdown' | 'management',\n"
    "  'expenses': [\n"
    "    {'amount': 500, 'category': 'groceries', 'note': null},\n"
    "    {'amount': 200, 'category': 'transport', 'note': 'Uber'}\n"
    "  ],\n"
    "  'query': null\n"
    "}\n"
    "(If intent is not expense_logging, set 'expenses' to null. If intent is not query, set 'query' to null.)"
)

def process_user_message(message: str) -> dict:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=512,
    )
    response = chat_completion.choices[0].message.content
    if not response:
        return {"intent": "unknown", "expenses": None, "query": None, "raw": ""}
    try:
        # Try to parse the response as JSON (replace single quotes with double quotes for safety)
        response_json = json.loads(response.replace("'", '"'))
        return response_json
    except Exception as e:
        print(f"[LLM Service] Failed to parse LLM response: {response}\nError: {e}")
        return {"intent": "unknown", "expenses": None, "query": None, "raw": response}

def generate_sql_from_query(user_message: str, user_id: int) -> str:
    sql_prompt = (
        f"You are an intelligent SQL agent for a personal expense tracker. "
        f"The user's data is in a PostgreSQL database with the following tables: users(id, phone_number, created_at), categories(id, name, user_id, is_custom), expenses(id, user_id, category_id, amount, timestamp, note). "
        f"The user's id is {user_id}. "
        f"When matching category names, use ILIKE for case-insensitive comparison. "
        f"Use COALESCE(SUM(amount), 0) to handle null values. "
        f"Format dates nicely using to_char(timestamp, 'Month DD, YYYY'). "
        f"Order results logically (most recent first, highest amounts first, etc.). "
        f"Limit results to reasonable numbers (5-10 rows max). "
        f"Use meaningful column aliases. "
        f"Given the user's question, generate a single SQL SELECT statement that answers it naturally and completely. "
        f"Respond ONLY with the SQL statement, no explanation."
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": user_message},
        ],
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=256,
    )
    sql = chat_completion.choices[0].message.content
    if sql:
        return sql.strip()
    return ""

def generate_breakdown_sql(user_id: int, time_period: str = "all") -> str:
    """Generate SQL for expense breakdown by category"""
    if time_period == "week":
        time_filter = "AND timestamp >= NOW() - INTERVAL '7 days'"
    elif time_period == "month":
        time_filter = "AND timestamp >= NOW() - INTERVAL '1 month'"
    else:
        time_filter = ""
    
    sql = f"""
    SELECT 
        c.name as category,
        SUM(e.amount) as total_amount,
        COUNT(*) as transaction_count
    FROM expenses e
    JOIN categories c ON e.category_id = c.id
    WHERE e.user_id = {user_id} {time_filter}
    GROUP BY c.name
    ORDER BY total_amount DESC
    """
    return sql.strip()

def format_breakdown_result(result, time_period: str = "all") -> str:
    """Format breakdown results into a user-friendly message"""
    if not result:
        period_text = "this period" if time_period != "all" else "any period"
        return f"No expenses found for {period_text}."
    
    total = sum(row.total_amount for row in result)
    period_text = {
        "all": "all time",
        "week": "this week", 
        "month": "this month"
    }.get(time_period, "this period")
    
    # Format breakdown with emojis
    emoji_map = {
        "transport": "🚗", "electronics": "💻", "lunch": "🍔", 
        "purchases": "🛒", "groceries": "🛍️", "entertainment": "🎬", 
        "health": "💊", "food": "🍕", "coffee": "☕", "shopping": "🛍️"
    }
    
    breakdown_lines = []
    for row in result:
        emoji = emoji_map.get(row.category.lower(), "💰")
        breakdown_lines.append(f"{emoji} {row.category.title()}: PKR {row.total_amount:,.0f} ({row.transaction_count} transactions)")
    
    breakdown_text = "\n".join(breakdown_lines)
    
    return f"📊 Your spending breakdown for {period_text}:\n\nTotal: PKR {total:,.0f}\n\n{breakdown_text}"

def format_summary_with_llm(summary_prompt: str) -> str:
    system_prompt = (
        "You are a financial assistant. When given a summary prompt, respond ONLY with a short, friendly, bullet-pointed WhatsApp message using emojis and line breaks. "
        "Do NOT return JSON or any structured data. Here is an example:\n"
        "Prompt: Weekly Expense Summary:\nTotal: PKR 10,000\nTop Categories:\n🚗 Transport: PKR 5,000\n🍔 Lunch: PKR 3,000\n🛒 Purchases: PKR 2,000\nBiggest single expense: PKR 5,000 (Transport)\nAverage per day: PKR 1,428\nRespond ONLY with a concise, bullet-pointed WhatsApp message using emojis and line breaks.\n"
        "Response: \n"
        "Your Weekly Summary 🗓️\nTotal: PKR 10,000\n• 🚗 Transport: PKR 5,000\n• 🍔 Lunch: PKR 3,000\n• 🛒 Purchases: PKR 2,000\nBiggest expense: PKR 5,000 (Transport)\nAvg/day: PKR 1,428\n"
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_prompt},
        ],
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=256,
    )
    response = chat_completion.choices[0].message.content
    if response:
        return response.strip()
    return ""
