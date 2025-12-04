#!/usr/bin/env python3
"""
Test script to verify the action button fix works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

# Mock RLCard to avoid import issues
import webapp.rlcard_mock as rlcard
from webapp.app import GameManager
from webapp.coach.llm_opponent_agent import LLMOpponentAgent

def test_action_processing():
    """Test that action 2 (Raise to 3BB) gets processed correctly"""

    print("🧪 Testing action processing fix...")

    # Create game manager
    game_manager = GameManager()

    # Start a new game
    session_id = game_manager.start_new_game()
    print(f"🎮 Started game session: {session_id}")

    # Get initial game state
    state = game_manager.get_game_state(session_id)
    print("📊 Initial game state retrieved")

    # Try to process action 2 (Raise to 3BB)
    print("🎯 Testing action 2 (Raise to 3BB)...")
    try:
        result = game_manager.process_action(session_id, 2)
        if result:
            print("✅ Action 2 processed successfully!")
            return True
        else:
            print("❌ Action 2 processing returned None")
            return False
    except Exception as e:
        print(f"❌ Action 2 processing failed: {e}")
        return False

if __name__ == "__main__":
    success = test_action_processing()
    sys.exit(0 if success else 1)
