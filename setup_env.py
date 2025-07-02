#!/usr/bin/env python3
"""
Setup script to configure environment variables for the Expense Bot
"""

import os
import sys

def create_env_file():
    """Create a .env file with the necessary configuration"""
    
    env_content = """# Expense Bot Environment Configuration
# =====================================

# Database Configuration
DATABASE_URL=sqlite:///./expense_bot.db

# Groq API Configuration
# Replace 'your-groq-api-key-here' with your actual Groq API key
GROQ_API_KEY=your-groq-api-key-here

# Intelligent Agent Configuration
USE_INTELLIGENT_AGENT=true
AGENT_USE_MEMORY=true
AGENT_USE_ADVANCED_SQL=true
AGENT_USE_NATURAL_RESPONSES=true
AGENT_LLM_MODEL=llama-3.3-70b-versatile
AGENT_TEMPERATURE=0.0
AGENT_MAX_TOKENS=1000
AGENT_MAX_HISTORY=10
AGENT_MEMORY_TTL=3600
AGENT_FALLBACK=true
AGENT_CONFIDENCE_THRESHOLD=0.7

# Redis Configuration (Optional - for enhanced memory)
# REDIS_URL=redis://localhost:6379

# Logging Configuration
DEBUG=true
LOG_LEVEL=INFO
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file successfully!")
        print("\n📝 Next steps:")
        print("1. Edit the .env file and replace 'your-groq-api-key-here' with your actual Groq API key")
        print("2. Get your API key from: https://console.groq.com/keys")
        print("3. Run: python3 test_groq_connection.py to test the connection")
        print("4. Run: python3 test_enhanced_agent.py to test the full system")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def check_groq_api_key():
    """Check if GROQ_API_KEY is set and valid"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        print("❌ GROQ_API_KEY not set or still using placeholder")
        return False
    
    print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")
    return True

if __name__ == "__main__":
    print("🔧 Expense Bot Environment Setup")
    print("=" * 40)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("📁 .env file already exists")
        # Load it to check the API key
        from dotenv import load_dotenv
        load_dotenv()
        
        if check_groq_api_key():
            print("\n🎉 Environment is configured!")
            print("You can now run: python3 test_enhanced_agent.py")
        else:
            print("\n⚠️  Please update your .env file with a valid Groq API key")
    else:
        print("📁 Creating .env file...")
        if create_env_file():
            print("\n📝 Please edit the .env file with your Groq API key")
        else:
            print("\n❌ Failed to create .env file")
            sys.exit(1) 