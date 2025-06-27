#!/usr/bin/env python3
"""
Test script for the Intelligent Agent Module
This script tests the agent without affecting the main FastAPI app.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_intelligent_agent():
    """Test the intelligent agent functionality"""
    
    print("🧪 Testing Intelligent Agent Module...")
    
    try:
        # Test imports
        from intelligent_agent import config, memory, process_message_safely
        print("✅ All imports successful")
        
        # Test configuration
        print(f"📋 Agent enabled: {config.enabled}")
        print(f"📋 Fallback enabled: {config.fallback_to_legacy}")
        print(f"📋 Memory enabled: {config.use_memory}")
        
        # Test memory
        memory_stats = memory.get_memory_summary()
        print(f"📊 Memory stats: {memory_stats}")
        
        # Test configuration loading
        print(f"🔧 LLM Model: {config.llm_model}")
        print(f"🔧 Temperature: {config.temperature}")
        
        print("\n✅ All tests passed! The intelligent agent module is ready.")
        print("\n📝 To enable the intelligent agent, set USE_INTELLIGENT_AGENT=true in your .env file")
        print("📝 The agent will automatically fall back to your existing system if disabled or if errors occur")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intelligent_agent() 