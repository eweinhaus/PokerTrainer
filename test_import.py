#!/usr/bin/env python3
"""
Simple test script to check if the imports work
"""
try:
    from webapp.coach.gto_agent import GTOAgent
    print("✓ GTOAgent import successful")
except Exception as e:
    print(f"✗ GTOAgent import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from webapp.coach.llm_opponent_agent import LLMOpponentAgent
    print("✓ LLMOpponentAgent import successful")
except Exception as e:
    print(f"✗ LLMOpponentAgent import failed: {e}")

try:
    from webapp.coach.chatbot_coach import ChatbotCoach
    print("✓ ChatbotCoach import successful")
except Exception as e:
    print(f"✗ ChatbotCoach import failed: {e}")

try:
    from webapp.coach.strategy_evaluator import StrategyEvaluator
    print("✓ StrategyEvaluator import successful")
except Exception as e:
    print(f"✗ StrategyEvaluator import failed: {e}")

