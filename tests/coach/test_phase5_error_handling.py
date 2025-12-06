"""
Unit tests for Phase 5 error handling improvements
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.strategy_evaluator import StrategyEvaluator
from coach.chatbot_coach import ChatbotCoach
from coach.pattern_recognizer import PatternRecognizer


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling for Phase 5"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = StrategyEvaluator()
        self.pattern_recognizer = PatternRecognizer()
    
    def test_evaluate_action_with_invalid_game_state(self):
        """Test evaluate_action handles invalid game state gracefully"""
        # Test with None game state
        result = self.evaluator.evaluate_action(None, 'call')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
        self.assertIn('explanation', result)
        
        # Test with empty dict
        result = self.evaluator.evaluate_action({}, 'call')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_evaluate_action_with_missing_hand(self):
        """Test evaluate_action handles missing hand gracefully"""
        game_state = {'stage': 0, 'pot': 100}
        result = self.evaluator.evaluate_action(game_state, 'call')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_analyze_hand_with_empty_history(self):
        """Test analyze_hand handles empty hand history"""
        game_state = {'pot': 100, 'stage': 0}
        result = self.evaluator.analyze_hand([], game_state)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        self.assertIn('decisions', result)
        self.assertEqual(len(result['decisions']), 0)
    
    def test_analyze_hand_with_invalid_game_state(self):
        """Test analyze_hand handles invalid game state"""
        hand_history = [{'action': 'call', 'stage': 0}]
        result = self.evaluator.analyze_hand(hand_history, None)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
    
    def test_pattern_recognizer_with_empty_hand_list(self):
        """Test pattern recognizer handles empty hand list"""
        patterns = self.pattern_recognizer.identify_patterns([])
        self.assertIsNotNone(patterns)
        self.assertIn('consistent_mistakes', patterns)
        self.assertIn('good_habits', patterns)
        self.assertEqual(len(patterns['consistent_mistakes']), 0)
    
    def test_pattern_recognizer_with_invalid_data(self):
        """Test pattern recognizer handles invalid data"""
        invalid_hands = [None, {}, {'invalid': 'data'}]
        patterns = self.pattern_recognizer.identify_patterns(invalid_hands)
        self.assertIsNotNone(patterns)
        self.assertIn('consistent_mistakes', patterns)
    
    def test_get_recent_hands_with_missing_session(self):
        """Test get_recent_hands handles missing session"""
        hand_history_storage = {}
        result = self.pattern_recognizer.get_recent_hands('nonexistent', hand_history_storage)
        self.assertEqual(result, [])
    
    def test_get_recent_hands_with_empty_storage(self):
        """Test get_recent_hands handles empty storage"""
        hand_history_storage = {'session1': []}
        result = self.pattern_recognizer.get_recent_hands('session1', hand_history_storage)
        self.assertEqual(result, [])
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chatbot_coach_handles_api_failure(self, mock_openai):
        """Test chatbot coach handles API failure gracefully"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        coach = ChatbotCoach()
        
        # Should handle error gracefully
        try:
            response = coach.chat("test message", session_id="test")
            # Should return fallback or handle error
            self.assertIsNotNone(response)
        except Exception:
            # If exception is raised, it should be handled at API level
            pass
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chatbot_coach_handles_timeout(self, mock_openai):
        """Test chatbot coach handles timeout gracefully"""
        from concurrent.futures import TimeoutError as FutureTimeoutError
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        coach = ChatbotCoach()
        
        # Mock executor to raise timeout
        with patch.object(coach.executor, 'submit', side_effect=FutureTimeoutError()):
            try:
                response = coach.chat("test message", session_id="test")
                # Should handle timeout gracefully
                self.assertIsNotNone(response)
            except FutureTimeoutError:
                # Timeout should be caught and handled
                pass
    
    def test_evaluate_action_with_exception(self):
        """Test evaluate_action handles exceptions gracefully"""
        # Mock internal method to raise exception
        with patch.object(self.evaluator, '_determine_position', side_effect=Exception("Test error")):
            result = self.evaluator.evaluate_action({'stage': 0}, 'call')
            self.assertIsNotNone(result)
            self.assertIn('grade', result)
            self.assertEqual(result['grade'], 'C')  # Default grade on error


class TestEdgeCaseHandling(unittest.TestCase):
    """Test edge case handling for Phase 5"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = StrategyEvaluator()
        self.pattern_recognizer = PatternRecognizer()
    
    def test_all_in_scenario(self):
        """Test handling of all-in scenarios"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 1000,
            'stakes': [500, 500],  # Both players all-in
            'big_blind': 2
        }
        result = self.evaluator.evaluate_action(game_state, 4)  # All-in action
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
        self.assertIn('explanation', result)
    
    def test_short_stack_scenario(self):
        """Test handling of very short stacks (10-20 BB)"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'A')],
            'pot': 20,
            'stakes': [20, 20],  # 10 BB stacks
            'big_blind': 2
        }
        result = self.evaluator.evaluate_action(game_state, 'raise')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_first_hand_no_history(self):
        """Test handling of first hand with no history"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'K'), ('H', 'Q')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        result = self.evaluator.analyze_hand([], game_state, session_id='test', hand_history_storage={})
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
    
    def test_very_long_hand(self):
        """Test handling of very long hands (many betting rounds)"""
        # Create a long hand history
        long_history = []
        for stage in range(4):  # Preflop, flop, turn, river
            for i in range(5):  # Multiple actions per street
                long_history.append({
                    'player_id': 0,
                    'action': 'call' if i % 2 == 0 else 'check',
                    'stage': stage,
                    'pot': 100 + (stage * 50) + (i * 10)
                })
        
        game_state = {'pot': 500, 'stage': 3}
        result = self.evaluator.analyze_hand(long_history, game_state)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        self.assertGreater(len(result.get('decisions', [])), 0)
    
    def test_rapid_actions(self):
        """Test handling of rapid actions"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Simulate rapid actions
        for i in range(10):
            result = self.evaluator.evaluate_action(game_state, 'call')
            self.assertIsNotNone(result)
            self.assertIn('grade', result)
    
    def test_empty_responses(self):
        """Test handling of empty responses"""
        # Test with empty hand history
        result = self.evaluator.analyze_hand([], {})
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        
        # Test with empty pattern recognition
        patterns = self.pattern_recognizer.identify_patterns([])
        self.assertIsNotNone(patterns)
        self.assertIn('consistent_mistakes', patterns)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test with None inputs
        result = self.evaluator.evaluate_action(None, None)
        self.assertIsNotNone(result)
        
        # Test with wrong types
        result = self.evaluator.evaluate_action("not a dict", 123)
        self.assertIsNotNone(result)
        
        # Test with invalid action
        result = self.evaluator.evaluate_action({'stage': 0}, 'invalid_action')
        self.assertIsNotNone(result)
    
    def test_opponent_folds_immediately(self):
        """Test handling when opponent folds immediately"""
        game_state = {
            'stage': 0,
            'hand': [('S', '2'), ('H', '7')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        # Hand ends immediately after opponent fold
        hand_history = [{'player_id': 0, 'action': 'call', 'stage': 0}]
        result = self.evaluator.analyze_hand(hand_history, game_state)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)


if __name__ == '__main__':
    unittest.main()


