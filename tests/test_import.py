#!/usr/bin/env python3
"""
Simple test script to check if the imports work
"""
try:
    from webapp.coach.gto_agent import GTOAgent
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    from webapp.coach.llm_opponent_agent import LLMOpponentAgent
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    from webapp.coach.chatbot_coach import ChatbotCoach
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    from webapp.coach.strategy_evaluator import StrategyEvaluator
except Exception as e:
    import traceback
    traceback.print_exc()

