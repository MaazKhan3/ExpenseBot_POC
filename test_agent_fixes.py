#!/usr/bin/env python3
"""
Test script to verify the fixes for the expense tracking agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.intelligent_agent_v3.agent_v3 import process_message_with_agent_v3
from app import crud, models

# Create test database
engine = create_engine("sqlite:///test_expense_agent_fixes.db")
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_agent_fixes():
    """Test the fixes for the expense tracking agent"""
    
    print("🧪 Testing Expense Agent Fixes")
    print("=" * 50)
    
    # Test cases to verify fixes
    test_cases = [
        {
            "name": "Fix 1: Incomplete expense - missing amount",
            "message": "bought a tennis racket",
            "expected_behavior": "Should ask for the amount spent on tennis racket",
            "phone": "923001234567"
        },
        {
            "name": "Fix 2: Incomplete expense - missing item",
            "message": "spent 500",
            "expected_behavior": "Should ask where the 500 was spent",
            "phone": "923001234567"
        },
        {
            "name": "Fix 3: Detailed success message",
            "message": "lunch 800",
            "expected_behavior": "Should show specific details of what was logged",
            "phone": "923001234567"
        },
        {
            "name": "Fix 4: Smart categorization",
            "message": "tennis racket 5000",
            "expected_behavior": "Should categorize as sports/entertainment, not generic shopping",
            "phone": "923001234567"
        },
        {
            "name": "Fix 5: Multiple expenses with notes",
            "message": "apples 500, mangoes 2k, oranges 600",
            "expected_behavior": "Should include notes for each item",
            "phone": "923001234567"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"📝 Message: '{test_case['message']}'")
        print(f"🎯 Expected: {test_case['expected_behavior']}")
        
        db = SessionLocal()
        try:
            # Process message
            response = process_message_with_agent_v3(
                phone_number=test_case['phone'],
                message=test_case['message'],
                db=db
            )
            
            print(f"📤 Response: {response['message']}")
            print(f"🎭 Intent: {response['intent']}")
            
            # Check if user exists and show their expenses
            user = crud.get_user_by_phone_number(db, test_case['phone'])
            if user:
                expenses = db.query(models.Expense).filter(models.Expense.user_id == user.id).all()
                if expenses:
                    print("💾 Stored expenses:")
                    for exp in expenses[-3:]:  # Show last 3 expenses
                        category = db.query(models.Category).filter(models.Category.id == exp.category_id).first()
                        print(f"   - {exp.amount} PKR ({category.name if category else 'unknown'}) - Note: '{exp.note}'")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            db.close()
        
        print("-" * 50)

if __name__ == "__main__":
    test_agent_fixes() 