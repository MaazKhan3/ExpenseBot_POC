#!/usr/bin/env python3
"""
Test to verify the context persistence fix
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
engine = create_engine("sqlite:///test_context_fix.db")
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_context_persistence():
    """Test that context persists between messages"""
    
    print("🧪 Testing Context Persistence Fix")
    print("=" * 50)
    
    # Test the exact scenario from the logs
    test_scenario = [
        {
            "message": "spent 800",
            "expected_response": "What did you spend 800 PKR on?",
            "expected_intent": "log_expense"
        },
        {
            "message": "popcorn",
            "expected_response": "✅ Logged 800 PKR for food (popcorn)",
            "expected_intent": "log_expense"
        }
    ]
    
    db = SessionLocal()
    try:
        for i, test_case in enumerate(test_scenario, 1):
            print(f"\n📝 Message {i}: '{test_case['message']}'")
            print(f"🎯 Expected Response: {test_case['expected_response']}")
            print(f"🎯 Expected Intent: {test_case['expected_intent']}")
            
            # Process message
            response = process_message_with_agent_v3(
                phone_number="923001234567",
                message=test_case['message'],
                db=db
            )
            
            print(f"📤 Actual Response: {response['message']}")
            print(f"🎭 Actual Intent: {response['intent']}")
            
            # Check if response matches expected
            if test_case['expected_response'] in response['message']:
                print("✅ Response matches expected pattern")
            else:
                print("❌ Response doesn't match expected pattern")
            
            if response['intent'] == test_case['expected_intent']:
                print("✅ Intent matches expected")
            else:
                print("❌ Intent doesn't match expected")
            
            # Check if user exists and show their expenses
            user = crud.get_user_by_phone_number(db, "923001234567")
            if user:
                expenses = db.query(models.Expense).filter(models.Expense.user_id == user.id).all()
                if expenses:
                    print("💾 Stored expenses:")
                    for exp in expenses[-3:]:  # Show last 3 expenses
                        category = db.query(models.Category).filter(models.Category.id == exp.category_id).first()
                        print(f"   - {exp.amount} PKR ({category.name if category else 'unknown'}) - Note: '{exp.note}'")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_context_persistence() 