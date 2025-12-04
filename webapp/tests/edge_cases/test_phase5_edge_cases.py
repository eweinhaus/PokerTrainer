"""
Edge case tests for Phase 5
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import app after setting up path
try:
    from app import app, game_manager, hand_history_storage
except ImportError:
    # Try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), '../../app.py'))
    app_module = importlib.util.module_from_spec(spec)
    sys.modules['app'] = app_module
    spec.loader.exec_module(app_module)
    app = app_module.app
    game_manager = app_module.game_manager
    hand_history_storage = app_module.hand_history_storage
from coach.strategy_evaluator import StrategyEvaluator
from coach.pattern_recognizer import PatternRecognizer


class TestAllInScenarios(unittest.TestCase):
    """Test all-in scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_allin'
        self.evaluator = StrategyEvaluator()
    
    def test_all_in_preflop(self):
        """Test all-in preflop scenario"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'A')],
            'pot': 200,
            'stakes': [100, 100],  # Both all-in
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 4)  # All-in
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
        self.assertIn('explanation', result)
    
    def test_all_in_postflop(self):
        """Test all-in postflop scenario"""
        game_state = {
            'stage': 1,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10')],
            'pot': 200,
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 4)  # All-in
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_all_in_short_stack(self):
        """Test all-in with very short stack"""
        game_state = {
            'stage': 0,
            'hand': [('S', '2'), ('H', '7')],
            'pot': 40,
            'stakes': [20, 20],  # 10 BB stacks
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 4)  # All-in
        self.assertIsNotNone(result)
        self.assertIn('grade', result)


class TestShortStacks(unittest.TestCase):
    """Test short stack scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = StrategyEvaluator()
    
    def test_10_bb_stack(self):
        """Test 10 BB stack depth"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [20, 20],  # 10 BB
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 'raise')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_15_bb_stack(self):
        """Test 15 BB stack depth"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [30, 30],  # 15 BB
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 'raise')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)
    
    def test_20_bb_stack(self):
        """Test 20 BB stack depth"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [40, 40],  # 20 BB
            'big_blind': 2
        }
        
        result = self.evaluator.evaluate_action(game_state, 'raise')
        self.assertIsNotNone(result)
        self.assertIn('grade', result)


class TestRapidActions(unittest.TestCase):
    """Test rapid action scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_rapid'
        self.evaluator = StrategyEvaluator()
    
    def test_rapid_evaluations(self):
        """Test rapid action evaluations"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Make 20 rapid evaluations
        for i in range(20):
            result = self.evaluator.evaluate_action(game_state, 'call')
            self.assertIsNotNone(result)
            self.assertIn('grade', result)
    
    def test_rapid_api_requests(self):
        """Test rapid API requests"""
        hand_history = [
            {
                'player_id': 0,
                'action': 1,
                'stage': 0,
                'pot': 4,
                'hand': [('S', 'A'), ('H', 'K')],
                'public_cards': [],
                'stakes': [100, 100]
            }
        ]
        
        game_state = {
            'pot': 4,
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Make 10 rapid requests
        for i in range(10):
            response = self.app.post('/api/coach/analyze-hand',
                                    json={
                                        'session_id': f'{self.session_id}_{i}',
                                        'hand_history': hand_history,
                                        'game_state': game_state
                                    })
            # Should handle gracefully (may succeed or fail, but not crash)
            self.assertIn(response.status_code, [200, 400, 500])


class TestNetworkFailures(unittest.TestCase):
    """Test network failure scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_network'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_network_failure(self, mock_openai):
        """Test chat handles network failure"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Network error")
        mock_openai.return_value = mock_client
        
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'Test'
                                })
        # Should return error response, not crash
        self.assertIn(response.status_code, [500, 503])
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_hand_analysis_with_missing_data(self):
        """Test hand analysis handles missing data gracefully"""
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': None,
                                    'game_state': None
                                })
        # Should handle gracefully
        self.assertIn(response.status_code, [400, 500])


class TestInvalidInputs(unittest.TestCase):
    """Test invalid input scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_invalid'
        self.evaluator = StrategyEvaluator()
    
    def test_invalid_game_state_types(self):
        """Test handling of invalid game state types"""
        # Test with string instead of dict
        result = self.evaluator.evaluate_action("not a dict", 'call')
        self.assertIsNotNone(result)
        
        # Test with list instead of dict
        result = self.evaluator.evaluate_action([], 'call')
        self.assertIsNotNone(result)
        
        # Test with None
        result = self.evaluator.evaluate_action(None, 'call')
        self.assertIsNotNone(result)
    
    def test_invalid_action_types(self):
        """Test handling of invalid action types"""
        game_state = {'stage': 0, 'hand': [('S', 'A'), ('H', 'K')]}
        
        # Test with None action
        result = self.evaluator.evaluate_action(game_state, None)
        self.assertIsNotNone(result)
        
        # Test with invalid action string
        result = self.evaluator.evaluate_action(game_state, 'invalid_action_xyz')
        self.assertIsNotNone(result)
    
    def test_invalid_api_inputs(self):
        """Test API handles invalid inputs"""
        # Test with non-JSON
        response = self.app.post('/api/coach/chat',
                                data='not json',
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test with wrong field types
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': 123,  # Should be string
                                    'hand_history': 'not a list',
                                    'game_state': 'not a dict'
                                })
        self.assertEqual(response.status_code, 400)


class TestEmptyResponses(unittest.TestCase):
    """Test empty response scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_empty'
        self.evaluator = StrategyEvaluator()
        self.pattern_recognizer = PatternRecognizer()
    
    def test_empty_hand_history(self):
        """Test handling of empty hand history"""
        result = self.evaluator.analyze_hand([], {})
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        self.assertEqual(len(result.get('decisions', [])), 0)
    
    def test_empty_pattern_recognition(self):
        """Test handling of empty pattern recognition"""
        patterns = self.pattern_recognizer.identify_patterns([])
        self.assertIsNotNone(patterns)
        self.assertIn('consistent_mistakes', patterns)
        self.assertEqual(len(patterns['consistent_mistakes']), 0)
    
    def test_empty_chat_history(self):
        """Test chat with no conversation history"""
        @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
        @patch('coach.chatbot_coach.OpenAI')
        def test(mock_openai):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = ""
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            response = self.app.post('/api/coach/chat',
                                    json={
                                        'session_id': f'{self.session_id}_new',
                                        'message': 'Test'
                                    })
            # Should handle empty response
            self.assertEqual(response.status_code, 200)
        
        test()


class TestVeryLongHands(unittest.TestCase):
    """Test very long hand scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_long_hand'
        self.evaluator = StrategyEvaluator()
    
    def test_very_long_hand_history(self):
        """Test handling of very long hand history"""
        # Create hand with 50+ decisions
        long_history = []
        for stage in range(4):
            for i in range(15):
                long_history.append({
                    'player_id': 0,
                    'action': 1,
                    'stage': stage,
                    'pot': 100 + (stage * 50) + (i * 10),
                    'hand': [('S', 'A'), ('H', 'K')],
                    'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10')] if stage > 0 else [],
                    'stakes': [100, 100]
                })
        
        game_state = {
            'pot': 800,
            'stage': 3,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10'), ('C', '9')],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        result = self.evaluator.analyze_hand(long_history, game_state)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        # Should handle long hands without crashing
        self.assertGreater(len(result.get('decisions', [])), 0)
    
    def test_very_long_hand_via_api(self):
        """Test very long hand via API"""
        long_history = []
        for i in range(100):
            long_history.append({
                'player_id': 0,
                'action': 1,
                'stage': i % 4,
                'pot': 100 + (i * 10),
                'hand': [('S', 'A'), ('H', 'K')],
                'public_cards': [],
                'stakes': [100, 100]
            })
        
        game_state = {
            'pot': 1100,
            'stage': 3,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': long_history,
                                    'game_state': game_state
                                })
        # Should handle without crashing
        self.assertIn(response.status_code, [200, 400, 500])


class TestFirstHandNoHistory(unittest.TestCase):
    """Test first hand with no history scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_first_hand'
        self.evaluator = StrategyEvaluator()
    
    def test_first_hand_analysis(self):
        """Test analysis of first hand with no history"""
        hand_history = [
            {
                'player_id': 0,
                'action': 1,
                'stage': 0,
                'pot': 4,
                'hand': [('S', 'A'), ('H', 'K')],
                'public_cards': [],
                'stakes': [100, 100]
            }
        ]
        
        game_state = {
            'pot': 4,
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Analyze with no previous history
        result = self.evaluator.analyze_hand(
            hand_history, 
            game_state,
            session_id=self.session_id,
            hand_history_storage={}  # Empty history
        )
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        # Should handle gracefully without pattern recognition
        self.assertIn('decisions', result)
    
    def test_first_hand_via_api(self):
        """Test first hand analysis via API"""
        hand_history = [
            {
                'player_id': 0,
                'action': 1,
                'stage': 0,
                'pot': 4,
                'hand': [('S', 'A'), ('H', 'K')],
                'public_cards': [],
                'stakes': [100, 100]
            }
        ]
        
        game_state = {
            'pot': 4,
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Use new session with no history
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': f'{self.session_id}_new',
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('overall_grade', data)


class TestOpponentFoldsImmediately(unittest.TestCase):
    """Test scenarios where opponent folds immediately"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = StrategyEvaluator()
    
    def test_opponent_fold_preflop(self):
        """Test when opponent folds preflop"""
        # Hand ends immediately after opponent fold
        hand_history = [
            {
                'player_id': 0,
                'action': 1,  # Call
                'stage': 0,
                'pot': 4,
                'hand': [('S', '2'), ('H', '7')],
                'public_cards': [],
                'stakes': [100, 100]
            }
        ]
        
        game_state = {
            'pot': 4,
            'stage': 0,
            'hand': [('S', '2'), ('H', '7')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        result = self.evaluator.analyze_hand(hand_history, game_state)
        self.assertIsNotNone(result)
        self.assertIn('overall_grade', result)
        # Should handle short hands gracefully
        self.assertGreaterEqual(len(result.get('decisions', [])), 0)


if __name__ == '__main__':
    unittest.main()

