#!/usr/bin/env python3
"""
Test script to trigger AI turn processing and see the new logging
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

# Mock the environment variables to avoid LLM errors
os.environ['LLM_PROVIDER'] = 'openai'
os.environ['OPENAI_API_KEY'] = 'test-key-placeholder'

from webapp.app import GameManager
from webapp.coach.llm_opponent_agent import LLMOpponentAgent

def test_ai_turn_logging():
    """Test AI turn processing with logging"""

    # Create game manager
    game_manager = GameManager()

    # Start a new game
    session_id = 'test_session_123'
    game_state = game_manager.start_game(session_id)


    # Process human action to trigger AI turn
    # Let's assume it's the human's turn first
    if game_state.get('current_player') == 0:
        try:
            # Try to fold (action 0)
            result = game_manager.process_action(session_id, 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return

    # Now it should be AI's turn
    if game_state.get('current_player') == 1 or (result and result.get('current_player') == 1):
        try:
            ai_result = game_manager.process_ai_turn(session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_ai_turn_logging()
