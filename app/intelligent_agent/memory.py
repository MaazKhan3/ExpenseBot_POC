"""
Conversation Memory Management for Intelligent Agent
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger("expensebot.intelligent_agent.memory")

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: float
    user_message: str
    bot_response: Optional[str] = None
    intent: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserContext:
    """User-specific conversation context"""
    phone_number: str
    name: Optional[str] = None
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=10))
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_interaction: float = field(default_factory=time.time)
    pending_expense: Optional[Dict[str, Any]] = None  # Track pending expense context
    
    def add_turn(self, user_message: str, bot_response: Optional[str] = None, 
                 intent: Optional[str] = None, confidence: float = 0.0) -> None:
        """Add a new conversation turn"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            confidence=confidence
        )
        self.conversation_history.append(turn)
        self.last_interaction = time.time()
    
    def get_recent_context(self, turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation context"""
        return list(self.conversation_history)[-turns:]
    
    def get_user_name(self) -> Optional[str]:
        """Extract or return user name"""
        return self.name
    
    def set_user_name(self, name: str) -> None:
        """Set user name"""
        self.name = name
        self.preferences["name"] = name
    
    def set_pending_expense(self, pending_expense: Optional[Dict[str, Any]]) -> None:
        """Set pending expense context"""
        self.pending_expense = pending_expense
    
    def get_pending_expense(self) -> Optional[Dict[str, Any]]:
        """Get pending expense context"""
        return self.pending_expense

class ConversationMemory:
    """Manages conversation memory across all users"""
    
    def __init__(self, max_history: int = 10, ttl_seconds: int = 3600):
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self.user_contexts: Dict[str, UserContext] = {}
        self._cleanup_timer = 0
    
    def get_user_context(self, phone_number: str) -> UserContext:
        """Get or create user context"""
        if phone_number not in self.user_contexts:
            self.user_contexts[phone_number] = UserContext(phone_number=phone_number)
        return self.user_contexts[phone_number]
    
    def add_conversation_turn(self, phone_number: str, user_message: str, 
                             bot_response: Optional[str] = None, intent: Optional[str] = None,
                             confidence: float = 0.0) -> None:
        """Add a conversation turn for a user"""
        context = self.get_user_context(phone_number)
        context.add_turn(user_message, bot_response, intent, confidence)
        
        # Extract user name if this is an introduction
        if intent == "introduction" and not context.name:
            name = self._extract_name_from_message(user_message)
            if name:
                context.set_user_name(name)
    
    def get_conversation_context(self, phone_number: str, turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation context for a user"""
        context = self.get_user_context(phone_number)
        return context.get_recent_context(turns)
    
    def get_pending_expense(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get pending expense for a user"""
        context = self.get_user_context(phone_number)
        return context.get_pending_expense()
    
    def set_pending_expense(self, phone_number: str, pending_expense: Optional[Dict[str, Any]]) -> None:
        """Set pending expense for a user"""
        context = self.get_user_context(phone_number)
        context.set_pending_expense(pending_expense)
    
    def get_user_preferences(self, phone_number: str) -> Dict[str, Any]:
        """Get user preferences"""
        context = self.get_user_context(phone_number)
        return context.preferences
    
    def update_user_preferences(self, phone_number: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        context = self.get_user_context(phone_number)
        context.preferences.update(preferences)
    
    def _extract_name_from_message(self, message: str) -> Optional[str]:
        """Extract name from introduction messages"""
        import re
        # Common patterns for name introduction
        patterns = [
            r"i am ([a-zA-Z]+)",
            r"i'm ([a-zA-Z]+)", 
            r"my name is ([a-zA-Z]+)",
            r"this is ([a-zA-Z]+)",
            r"call me ([a-zA-Z]+)"
        ]
        
        message_lower = message.lower()
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1).title()
        return None
    
    def cleanup_expired_contexts(self) -> None:
        """Remove expired user contexts"""
        current_time = time.time()
        expired_users = []
        
        for phone_number, context in self.user_contexts.items():
            if current_time - context.last_interaction > self.ttl_seconds:
                expired_users.append(phone_number)
        
        for phone_number in expired_users:
            del self.user_contexts[phone_number]
            logger.info(f"Cleaned up expired context for user {phone_number}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        return {
            "total_users": len(self.user_contexts),
            "max_history": self.max_history,
            "ttl_seconds": self.ttl_seconds
        }

# Global memory instance
memory = ConversationMemory() 