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
    "Given a user's WhatsApp message, classify the intent as one of: 'expense_logging', 'query', or 'management'. "
    "If intent is 'expense_logging', extract all (amount, category, note) tuples from the message. "
    "If intent is 'query', extract the query type and any relevant time ranges or categories. "
    "Respond ONLY in the following JSON format:\n"
    "{\n"
    "  'intent': 'expense_logging' | 'query' | 'management',\n"
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
        f"You are an SQL agent for a personal expense tracker. "
        f"The user's data is in a PostgreSQL database with the following tables: users(id, phone_number, created_at), categories(id, name, user_id, is_custom), expenses(id, user_id, category_id, amount, timestamp, note). "
        f"The user's id is {user_id}. "
        f"Given the user's question, generate a single SQL SELECT statement that answers it, filtering by user_id where appropriate. "
        f"Respond ONLY with the SQL statement, no explanation."
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": user_message},
        ],
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=256,
    )
    sql = chat_completion.choices[0].message.content
    if sql:
        return sql.strip()
    return ""
