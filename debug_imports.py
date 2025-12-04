#!/usr/bin/env python3
"""
Debug script to test imports individually
"""
import sys
import os

# Add webapp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

print("Testing imports...")

# Test 1: Basic imports
try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")

try:
    import rlcard
    print("✓ rlcard imported")
except Exception as e:
    print(f"✗ rlcard failed: {e}")

# Test 2: Coach imports
try:
    from coach.gto_rules import GTORules
    print("✓ GTORules imported")
except Exception as e:
    print(f"✗ GTORules failed: {e}")

try:
    from coach.gto_agent import GTOAgent
    print("✓ GTOAgent imported")
except Exception as e:
    print(f"✗ GTOAgent failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from coach.strategy_evaluator import StrategyEvaluator
    print("✓ StrategyEvaluator imported")
except Exception as e:
    print(f"✗ StrategyEvaluator failed: {e}")

try:
    from coach.llm_opponent_agent import LLMOpponentAgent
    print("✓ LLMOpponentAgent imported")
except Exception as e:
    print(f"✗ LLMOpponentAgent failed: {e}")

try:
    from coach.chatbot_coach import ChatbotCoach
    print("✓ ChatbotCoach imported")
except Exception as e:
    print(f"✗ ChatbotCoach failed: {e}")

print("Import testing complete.")

