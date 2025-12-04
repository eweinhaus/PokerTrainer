#!/usr/bin/env python3
"""
Verify the app can start and all imports work
"""
import sys
import os

# Add webapp to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

print("Testing app startup...")

try:
    # Test all the critical imports
    from coach.gto_agent import GTOAgent
    print("✓ GTOAgent imported successfully")

    from coach.llm_opponent_agent import LLMOpponentAgent
    print("✓ LLMOpponentAgent imported successfully")

    from coach.chatbot_coach import ChatbotCoach
    print("✓ ChatbotCoach imported successfully")

    from coach.strategy_evaluator import StrategyEvaluator
    print("✓ StrategyEvaluator imported successfully")

    # Try to import the main app
    from app import app, game_manager
    print("✓ Flask app imported successfully")

    print("\n🎉 All imports successful! The app should start properly.")
    print("Try running: cd webapp && python app.py")
    print("Then visit: http://localhost:5001")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

