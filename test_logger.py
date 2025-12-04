#!/usr/bin/env python3
"""
Test logger fix
"""
import sys
import os

# Add webapp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

try:
    from coach.gto_agent import GTOAgent
    agent = GTOAgent(5)  # Create agent with 5 actions

    # Test a simple state to see if logger works
    test_state = {
        'raw_obs': {
            'hand': [0, 13],  # Some cards
            'stage': 0,       # Preflop
            'raised': [2, 0], # SB raised 2, BB raised 0
            'pot': 3,
            'big_blind': 2,
            'all_chips': [100, 100]
        },
        'raw_legal_actions': [0, 1, 2, 3, 4]  # All actions legal
    }

    # This should not cause logger error
    action = agent.step(test_state)
    print(f"✓ GTOAgent.step() completed successfully, returned action: {action}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

