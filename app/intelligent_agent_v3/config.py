#config.py

import os
from dataclasses import dataclass

@dataclass
class AgentConfigV2:
    """Configuration for the intelligent agent"""
    
    # LLM Configuration
    llm_model: str = "llama-3.3-70b-versatile"  # Reverted to previous model
    temperature: float = 0 # Lower temperature for more consistent responses
    max_tokens: int = 1000
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Agent Behavior
    enable_conversation_memory: bool = True
    max_conversation_history: int = 10
    
    # Database
    enable_advanced_queries: bool = True
    
    # Logging
    debug_mode: bool = True

# Global configuration instance
config = AgentConfigV2() 