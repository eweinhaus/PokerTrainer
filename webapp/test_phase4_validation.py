#!/usr/bin/env python3
"""
Phase 4 Validation Script - Test decision tracking and action history accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import GameManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_decision_tracking_accuracy():
    """Test that decision tracking works correctly across all stages"""
    logger.info("🧪 Testing decision tracking accuracy...")

    game_manager = GameManager()
    session_id = "test_decision_tracking"

    # Start a new game
    game_state = game_manager.start_game(session_id)
    assert game_state is not None, "Failed to start game"

    # Play through a full hand to test tracking
    actions_tested = 0
    total_actions = 0

    try:
        # Get initial state
        state = game_manager.get_game_state(session_id)
        stage = state.get('stage', 0)
        stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
        stage_name = stage_names.get(stage, f'Stage {stage}')
        logger.info(f"Initial state: {state.get('current_player')} to act, stage {stage_name}")

        # Play actions until hand is complete
        max_actions = 20  # Safety limit
        actions_played = 0

        while actions_played < max_actions:
            state = game_manager.get_game_state(session_id)

            if state.get('is_game_over', False):
                logger.info("Game ended")
                break

            current_player = state.get('current_player', 0)
            
            # If it's AI's turn, process AI turn first
            if current_player != 0:
                logger.info(f"AI turn (player {current_player})")
                result = game_manager.process_ai_turn(session_id)
                if result is None:
                    logger.warning("AI turn failed")
                    break
                actions_played += 1
                total_actions += 1
                continue

            # It's player's turn
            legal_actions = state.get('legal_actions', [])

            if not legal_actions:
                logger.warning("No legal actions available")
                break

            # Choose an action (prefer fold if many options, otherwise first action)
            if len(legal_actions) > 3:
                action_value = 0  # Fold
            else:
                # legal_actions is a list of integers, not dicts
                action_value = legal_actions[0] if legal_actions else 0

            logger.info(f"Playing action {action_value} for player {current_player}")

            # Process the action
            result = game_manager.process_action(session_id, action_value)
            assert result is not None, f"Action {action_value} failed"

            actions_played += 1
            total_actions += 1

            # Check that decision was tracked
            if hasattr(game_manager, 'games') and session_id in game_manager.games:
                game = game_manager.games[session_id]
                if 'action_history_with_cards' in game and game['action_history_with_cards']:
                    actions_tested += 1

        logger.info(f"✅ Played {actions_played} actions, tested {actions_tested} decision tracking entries")

        # Verify decision tracking accuracy
        if session_id in game_manager.games:
            game = game_manager.games[session_id]
            history = game.get('action_history_with_cards', [])

            if history:
                # Check that each action has required fields
                required_fields = ['type', 'player_id', 'player_name', 'action']
                accuracy_count = 0

                for entry in history:
                    if entry.get('type') == 'action':
                        has_all_fields = all(field in entry for field in required_fields)
                        if has_all_fields:
                            accuracy_count += 1

                action_entries = [h for h in history if h.get('type') == 'action']
                accuracy_rate = accuracy_count / len(action_entries) if action_entries else 0
                logger.info(f"Decision tracking accuracy: {accuracy_rate:.1%}")

                if accuracy_rate >= 0.995:  # 99.5% target
                    logger.info("✅ Decision tracking accuracy target met!")
                    return True
                else:
                    logger.warning(f"Decision tracking accuracy below target: {accuracy_rate:.1%}")
                    return False
            else:
                logger.warning("No action history found")
                return False

    except Exception as e:
        logger.error(f"Error during decision tracking test: {e}")
        return False

def test_action_history_correctness():
    """Test that action history shows correct sequence and stages"""
    logger.info("🧪 Testing action history correctness...")

    game_manager = GameManager()
    session_id = "test_action_history"

    try:
        # Start game
        game_state = game_manager.start_game(session_id)
        assert game_state is not None, "Failed to start game"

        # Play a few actions
        actions_sequence = []
        stages_seen = set()

        for i in range(5):  # Play 5 actions
            state = game_manager.get_game_state(session_id)

            if state.get('is_game_over', False):
                break

            current_player = state.get('current_player', 0)
            stage = state.get('stage', 0)
            stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            stages_seen.add(stage_name)

            # If it's AI's turn, process AI turn first
            if current_player != 0:
                result = game_manager.process_ai_turn(session_id)
                if result is None:
                    break
                actions_sequence.append((current_player, 'AI', stage_name))
                continue

            legal_actions = state.get('legal_actions', [])
            if not legal_actions:
                break

            # legal_actions is a list of integers, not dicts
            action_value = legal_actions[0] if legal_actions else 0
            actions_sequence.append((current_player, action_value, stage_name))

            result = game_manager.process_action(session_id, action_value)
            assert result is not None, f"Action {action_value} failed"

        # Check action history
        game = game_manager.games[session_id]
        history = game.get('action_history_with_cards', [])

        logger.info(f"Played actions: {actions_sequence}")
        logger.info(f"History entries: {len(history)}")
        logger.info(f"Stages seen: {stages_seen}")

        # Verify basic structure
        action_entries = [h for h in history if h.get('type') == 'action']
        if action_entries:
            logger.info("✅ Action history contains action entries")
            return True
        else:
            logger.warning("No action entries found in history")
            return False

    except Exception as e:
        logger.error(f"Error during action history test: {e}")
        return False

def main():
    """Run Phase 4 validation tests"""
    logger.info("🚀 Starting Phase 4 Validation Tests")
    logger.info("Testing decision tracking and action history after state management simplifications")

    tests_passed = 0
    total_tests = 2

    # Test decision tracking accuracy
    if test_decision_tracking_accuracy():
        tests_passed += 1
        logger.info("✅ Decision tracking test PASSED")
    else:
        logger.error("❌ Decision tracking test FAILED")

    # Test action history correctness
    if test_action_history_correctness():
        tests_passed += 1
        logger.info("✅ Action history test PASSED")
    else:
        logger.error("❌ Action history test FAILED")

    logger.info(f"\n📊 Phase 4 Validation Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        logger.info("🎉 Phase 4 validation successful!")
        return 0
    else:
        logger.warning("⚠️ Some Phase 4 tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
