#!/usr/bin/env python3
"""
Verify the app can start and all imports work
"""
import sys
import os

# Add webapp to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))


try:
    # Test all the critical imports
    from coach.gto_agent import GTOAgent

    from coach.llm_opponent_agent import LLMOpponentAgent

    from coach.chatbot_coach import ChatbotCoach

    from coach.strategy_evaluator import StrategyEvaluator

    # Try to import the main app
    from app import app, game_manager


except ImportError as e:
    sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

