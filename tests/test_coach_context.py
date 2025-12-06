#!/usr/bin/env python3
"""
Test script to verify AI coach context injection includes opponent actions
"""

import sys
import os
sys.path.append('webapp')

from coach.chatbot_coach import ChatbotCoach

def test_coach_context_injection():
    """Test that coach context injection includes opponent actions"""

    # Create coach instance
    coach = ChatbotCoach()

    # Mock hand history with both player and opponent actions
    # Based on the terminal logs: P0 raises, P1 3-bets, P0 folds
    hand_history = [
        {
            'player_id': 0,
            'action': 3,  # Raise Pot
            'stage': 0,   # Preflop
            'hand': ['D6', 'D7'],
            'public_cards': [],
            'position': 'button'
        },
        {
            'player_id': 1,
            'action': 2,  # Raise Half Pot (3-bet)
            'stage': 0,   # Preflop
            'hand': ['DJ', 'HJ'],
            'public_cards': [],
            'position': 'big_blind'
        },
        {
            'player_id': 0,
            'action': 0,  # Fold
            'stage': 0,   # Preflop
            'hand': ['D6', 'D7'],
            'public_cards': ['CK', 'DQ', 'H3', 'HA', 'C6'],
            'position': 'button'
        }
    ]

    # Mock current game context
    game_context = {
        'hand': ['D6', 'D7'],
        'public_cards': ['CK', 'DQ', 'H3', 'HA', 'C6'],
        'pot': 9,
        'position': 'big_blind'
    }

    # Test context injection
    message = "Was that a good fold or should I have 4bet?"
    print(f"Input hand_history: {len(hand_history)} decisions")
    for i, d in enumerate(hand_history):
        print(f"  Decision {i}: player_id={d.get('player_id')}, action={d.get('action')}, stage={d.get('stage')}")

    context = coach._inject_context(game_context, hand_history, message)

    print("\n=== Context Injection Test ===")
    print(f"Message: {message}")
    print(f"Context length: {len(context)} characters")
    print("\nContext content:")
    print(context)

    # Check if opponent action is included
    if "Opp" in context and "Raise" in context:
        print("\n✅ SUCCESS: Opponent actions are included in context!")
        return True
    else:
        print("\n❌ FAILURE: Opponent actions are missing from context!")
        return False

if __name__ == "__main__":
    success = test_coach_context_injection()
    sys.exit(0 if success else 1)
