"""
Configuration for the Intelligent ExpenseBot Agent
"""

import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration for the intelligent agent"""
    
    # Feature flags
    enabled: bool = False
    use_memory: bool = True
    use_advanced_sql: bool = True
    use_natural_responses: bool = True
    
    # LLM Configuration
    llm_model: str = "llama3-8b-8192"  # Your current model
    temperature: float = 0.0
    max_tokens: int = 512
    
    # Memory Configuration
    max_conversation_history: int = 10
    memory_ttl_seconds: int = 3600  # 1 hour
    
    # Fallback Configuration
    fallback_to_legacy: bool = True
    confidence_threshold: float = 0.7
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables"""
        return cls(
            enabled=os.getenv("USE_INTELLIGENT_AGENT", "false").lower() == "true",
            use_memory=os.getenv("AGENT_USE_MEMORY", "true").lower() == "true",
            use_advanced_sql=os.getenv("AGENT_USE_ADVANCED_SQL", "true").lower() == "true",
            use_natural_responses=os.getenv("AGENT_USE_NATURAL_RESPONSES", "true").lower() == "true",
            llm_model=os.getenv("AGENT_LLM_MODEL", "llama3-8b-8192"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "512")),
            max_conversation_history=int(os.getenv("AGENT_MAX_HISTORY", "10")),
            memory_ttl_seconds=int(os.getenv("AGENT_MEMORY_TTL", "3600")),
            fallback_to_legacy=os.getenv("AGENT_FALLBACK", "true").lower() == "true",
            confidence_threshold=float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.7"))
        )

# Global config instance
config = AgentConfig.from_env() 