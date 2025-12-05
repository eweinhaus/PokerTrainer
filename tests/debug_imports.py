#!/usr/bin/env python3
"""
Debug script to test imports individually
"""
import sys
import os

# Add webapp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))


# Test 1: Basic imports
try:
    import numpy as np
except Exception as e:

try:
    import rlcard
except Exception as e:

# Test 2: Coach imports
try:
    from coach.gto_rules import GTORules
except Exception as e:

try:
    from coach.gto_agent import GTOAgent
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    from coach.strategy_evaluator import StrategyEvaluator
except Exception as e:

try:
    from coach.llm_opponent_agent import LLMOpponentAgent
except Exception as e:

try:
    from coach.chatbot_coach import ChatbotCoach
except Exception as e:


