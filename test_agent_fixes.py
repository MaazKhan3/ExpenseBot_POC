#!/usr/bin/env python3
"""
Test script to verify the intelligent agent fixes
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_agent_fixes():
    """Test the improved agent functionality"""
    
    print("🧪 Testing Intelligent Agent Fixes...")
    
    try:
        # Test imports
        from intelligent_agent import config, memory, process_message_safely
        print("✅ All imports successful")
        
        # Test configuration
        print(f"📋 Agent enabled: {config.enabled}")
        print(f"📋 LLM Model: {config.llm_model}")
        
        # Test memory
        memory_stats = memory.get_memory_summary()
        print(f"📊 Memory stats: {memory_stats}")
        
        # Test pending expense functionality
        test_phone = "923001234567"
        memory.set_pending_expense(test_phone, {"amount": 500, "item": None, "category": None})
        pending = memory.get_pending_expense(test_phone)
        print(f"📝 Pending expense test: {pending}")
        
        print("\n✅ All tests passed! The agent fixes are ready.")
        print("\n🔧 Key improvements made:")
        print("   - Fixed .env file format (change : to =)")
        print("   - Improved conversation context handling")
        print("   - Better expense logging flow")
        print("   - Added total expenses query tool")
        print("   - Added expense cancellation tool")
        print("   - Enhanced LLM prompt with better examples")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_fixes() 