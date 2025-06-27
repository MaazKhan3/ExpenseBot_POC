"""
Intelligent ExpenseBot Agent (LangGraph/LangChain)
This module implements the next-gen, context-aware, multi-turn conversational agent for expense tracking.
"""

# Placeholder imports for LangChain, LangGraph, tools, memory, etc.
# from langchain.llms import ...
# from langgraph.graph import ...
# from langchain.memory import ...
# from langchain.tools import ...

import logging
from datetime import datetime
from collections import defaultdict, deque
import re

logger = logging.getLogger("expensebot.agent")
logger.setLevel(logging.INFO)

# In-memory short-term context (per user)
SESSION_CONTEXT = defaultdict(lambda: deque(maxlen=5))  # Last 5 messages per user

class IntelligentExpenseAgent:
    """
    Modular, extensible agent for ExpenseBot with logging, intent detection, chitchat, and context.
    """
    def __init__(self, db):
        self.db = db
        self.session_context = SESSION_CONTEXT

    def detect_intent(self, message):
        """Very basic intent detection for demo; replace with LLM or LangChain later."""
        msg = message.lower().strip()
        if any(greet in msg for greet in ["hi", "hello", "hey", "salaam", "assalam"]):
            return "greeting"
        if any(kw in msg for kw in ["thank", "thanks", "shukriya"]):
            return "thanks"
        if any(kw in msg for kw in ["joke", "funny", "laugh"]):
            return "joke"
        if any(kw in msg for kw in ["again", "repeat", "previous", "last time"]):
            return "repeat"
        # Detect introductions (e.g., "I am Maaz", "My name is Maaz", "I'm Maaz")
        if re.search(r"\b(i am|i'm|my name is|this is)\b", msg):
            return "introduction"
        # Add more rules or LLM call here
        return "other"

    def handle_chitchat(self, intent, message, phone_number):
        if intent == "greeting":
            return "Hello! 👋 How can I help you with your expenses today?"
        if intent == "thanks":
            return "You're welcome! 😊 Let me know if you need anything else."
        if intent == "joke":
            return "Why did the wallet go to therapy? It lost its sense of balance! 😄"
        if intent == "repeat":
            # Use context to repeat last bot message
            history = self.session_context[phone_number]
            for msg in reversed(history):
                if msg["sender"] == "bot":
                    return f"Here's what I said earlier: {msg['text']}"
            return "I don't have anything recent to repeat, but I'm here to help!"
        if intent == "introduction":
            # Try to extract the user's name
            match = re.search(r"(?:i am|i'm|my name is|this is)\s+([a-zA-Z]+)", message, re.IGNORECASE)
            if match:
                name = match.group(1)
                return f"Nice to meet you, {name}! 😊 I'm here to help you track your expenses."
            return "Nice to meet you! 😊 I'm here to help you track your expenses."
        return None

    def update_context(self, phone_number, sender, text):
        self.session_context[phone_number].append({"sender": sender, "text": text, "timestamp": datetime.utcnow().isoformat()})

    def run(self, message_body, phone_number, timestamp):
        logger.info(f"Received message from {phone_number} at {timestamp}: {message_body}")
        self.update_context(phone_number, "user", message_body)
        intent = self.detect_intent(message_body)
        logger.info(f"Detected intent: {intent}")
        chitchat_response = self.handle_chitchat(intent, message_body, phone_number)
        if chitchat_response:
            logger.info(f"Chitchat response: {chitchat_response}")
            self.update_context(phone_number, "bot", chitchat_response)
            return {"message": chitchat_response}
        # TODO: Add expense logging, query, and memory logic here
        logger.info("No chitchat matched; passing to main bot logic.")
        return None

def run_intelligent_agent(message_body, phone_number, timestamp, db):
    agent = IntelligentExpenseAgent(db)
    response = agent.run(message_body, phone_number, timestamp)
    if response:
        return response
    # Fallback: return None so main.py can use original logic
    return None 