#!/usr/bin/env python3
"""
Debug script to check environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Environment Variable Debug:")
print(f"USE_INTELLIGENT_AGENT: {os.getenv('USE_INTELLIGENT_AGENT')}")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY', 'NOT_SET')[:10]}...")

# Test the intelligent agent config
try:
    from app.intelligent_agent.config import config
    print(f"\n🤖 Intelligent Agent Config:")
    print(f"Enabled: {config.enabled}")
    print(f"Fallback: {config.fallback_to_legacy}")
    print(f"Memory: {config.use_memory}")
except Exception as e:
    print(f"\n❌ Error loading config: {e}") 