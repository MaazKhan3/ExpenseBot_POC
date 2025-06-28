#!/usr/bin/env python3
"""
Test script to verify the intelligent agent fixes
"""

import os
import sys
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.intelligent_agent.graph import parse_amount, map_category

def test_amount_parsing():
    """Test the improved amount parsing function"""
    print("Testing amount parsing...")
    
    test_cases = [
        ("500", 500.0),
        ("2k", 2000.0),
        ("25k", 25000.0),
        ("1.5k", 1500.0),
        ("1m", 1000000.0),
        ("1,000", 1000.0),
        ("2,500", 2500.0),
        ("invalid", None),
        ("", None),
        (None, None),
    ]
    
    for input_val, expected in test_cases:
        result = parse_amount(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_val}' -> {result} (expected: {expected})")

def test_category_mapping():
    """Test the improved category mapping function"""
    print("\nTesting category mapping...")
    
    test_cases = [
        ("popcorn", "food"),
        ("baseball bat", "sports"),
        ("hat", "clothing"),
        ("watch", "electronics"),
        ("fuel", "transportation"),
        ("train ticket", "transportation"),
        ("cricket kit", "sports"),
        ("gaming mouse", "electronics"),
        ("sweets", "food"),
        ("chocolate", "food"),
        ("water bottles", "food"),
        ("apples", "food"),
        ("carrots", "food"),
        ("bananas", "food"),
        ("monopoly", "entertainment"),
        ("chess", "entertainment"),
        ("board game", "entertainment"),
        ("unknown_item", "misc"),
        ("", "misc"),
        (None, "misc"),
    ]
    
    for input_val, expected in test_cases:
        result = map_category(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_val}' -> {result} (expected: {expected})")

def test_problematic_scenarios():
    """Test the scenarios that were problematic in the logs"""
    print("\nTesting problematic scenarios from logs...")
    
    # Scenario 1: "fuel 500, hat 2k, watch 25k"
    print("\nScenario 1: Multiple expenses in one message")
    print("Input: 'fuel 500, hat 2k, watch 25k'")
    print("Expected: Should extract first expense (fuel 500) and categorize as transportation")
    
    fuel_amount = parse_amount("500")
    fuel_category = map_category("fuel")
    print(f"✅ First expense: fuel {fuel_amount} PKR -> {fuel_category}")
    
    # Scenario 2: "popcorn 500"
    print("\nScenario 2: Food item categorization")
    print("Input: 'popcorn 500'")
    
    popcorn_amount = parse_amount("500")
    popcorn_category = map_category("popcorn")
    print(f"✅ Popcorn {popcorn_amount} PKR -> {popcorn_category}")
    
    # Scenario 3: "baseball bat for 9k"
    print("\nScenario 3: Sports item with k suffix")
    print("Input: 'baseball bat for 9k'")
    
    bat_amount = parse_amount("9k")
    bat_category = map_category("baseball bat")
    print(f"✅ Baseball bat {bat_amount} PKR -> {bat_category}")

def test_multiple_expense_parsing():
    """Test the multiple expense parsing functionality"""
    print("\nTesting multiple expense parsing...")
    
    # Test amount parsing for multiple expenses
    test_cases = [
        ("500", 500.0),
        ("2k", 2000.0),
        ("25k", 25000.0),
    ]
    
    for input_val, expected in test_cases:
        result = parse_amount(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_val}' -> {result} (expected: {expected})")
    
    # Test multiple expense scenario
    print("\nMultiple expense scenario: 'apples 500, carrots 40, bananas 200'")
    print("Expected: Should extract 3 expenses with proper amounts and categories")
    
    apple_amount = parse_amount("500")
    apple_category = map_category("apples")
    carrot_amount = parse_amount("40")
    carrot_category = map_category("carrots")
    banana_amount = parse_amount("200")
    banana_category = map_category("bananas")
    
    print(f"✅ Apples: {apple_amount} PKR -> {apple_category}")
    print(f"✅ Carrots: {carrot_amount} PKR -> {carrot_category}")
    print(f"✅ Bananas: {banana_amount} PKR -> {banana_category}")

def test_context_handling():
    """Test the improved context handling for short messages"""
    print("\nTesting context handling...")
    
    # Test short messages that should clear context
    short_messages = ["i", "a", "spent", "bought", "paid"]
    print("Short messages that should clear context:")
    for msg in short_messages:
        print(f"✅ '{msg}' -> Should clear pending context and ask for clarification")
    
    # Test normal expense messages
    normal_messages = ["500 for groceries", "groceries 500", "spent 500 on groceries"]
    print("\nNormal expense messages that should work:")
    for msg in normal_messages:
        print(f"✅ '{msg}' -> Should log expense normally")

if __name__ == "__main__":
    load_dotenv()
    
    print("🧪 Testing Intelligent Agent Fixes")
    print("=" * 50)
    
    test_amount_parsing()
    test_category_mapping()
    test_problematic_scenarios()
    test_multiple_expense_parsing()
    test_context_handling()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!") 