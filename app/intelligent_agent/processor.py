"""
Safe Integration Processor for Intelligent Agent
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.intelligent_agent.config import config
from app.intelligent_agent.graph import process_message_with_agent
from app.intelligent_agent.memory import memory

logger = logging.getLogger("expensebot.intelligent_agent.processor")

def process_message_safely(phone_number: str, message: str, db) -> Optional[Dict[str, Any]]:
    """
    Safely process a message with the intelligent agent.
    Returns None if agent is disabled or encounters an error.
    """
    try:
        # Check if agent is enabled
        if not config.enabled:
            logger.info("🤖 Agent is disabled, returning None")
            return None
        
        # Get user from database
        from ..models import User
        user = db.query(User).filter(User.phone_number == phone_number).first()
        if not user:
            logger.warning(f"🤖 User not found for phone number: {phone_number}")
            return None
        
        # Get conversation context
        conversation_history = memory.get_conversation_context(phone_number)
        pending_expense = memory.get_pending_expense(phone_number)
        
        logger.info(f"🤖 Processing message: {message}")
        logger.info(f"🤖 Conversation history length: {len(conversation_history)}")
        logger.info(f"🤖 Pending expense: {pending_expense}")
        
        # Process with agent
        result = process_message_with_agent(phone_number, message, db)
        
        if result:
            logger.info(f"🤖 Agent result: {result}")
            return result
        else:
            logger.info("🤖 Agent returned None, falling back to legacy system")
            return None
            
    except Exception as e:
        logger.error(f"🤖 Error in process_message_safely: {e}")
        import traceback
        traceback.print_exc()
        return None

def cleanup_memory():
    """Clean up expired conversation memory"""
    try:
        memory.cleanup_expired_contexts()
        logger.debug("Memory cleanup completed")
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")

def get_memory_stats() -> Dict[str, Any]:
    """Get memory usage statistics"""
    try:
        return memory.get_memory_summary()
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {"error": "Failed to get memory statistics"} 