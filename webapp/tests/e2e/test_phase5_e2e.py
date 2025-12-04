"""
End-to-end tests for Phase 5 critical user paths
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
import json
import time

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


class TestHandAnalysisE2E(unittest.TestCase):
    """Test complete hand analysis flow end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_e2e_hand_analysis'
    
    def test_complete_hand_analysis_workflow(self):
        """Test complete workflow: play hand → receive analysis"""
        # Step 1: Start a game
        response = self.app.post('/api/game/start',
                                json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 200)
        initial_state = json.loads(response.data)
        self.assertIn('hand', initial_state)
        
        # Step 2: Play a hand (simulate actions)
        # Player action
        response = self.app.post('/api/game/action',
                                json={
                                    'session_id': self.session_id,
                                    'action_index': 1  # Call
                                })
        self.assertEqual(response.status_code, 200)
        
        # AI turn
        response = self.app.post('/api/game/ai-turn',
                                json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 200)
        
        # Step 3: Get hand history
        response = self.app.get(f'/api/coach/get-hand-history?session_id={self.session_id}')
        self.assertEqual(response.status_code, 200)
        history_data = json.loads(response.data)
        self.assertIn('decisions', history_data)
        
        # Step 4: Analyze hand
        hand_history = history_data.get('decisions', [])
        game_state = {
            'pot': 4,
            'stage': 0,
            'hand': initial_state.get('hand', []),
            'public_cards': initial_state.get('public_cards', []),
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        self.assertEqual(response.status_code, 200)
        analysis = json.loads(response.data)
        
        # Step 5: Verify analysis contains all required fields
        self.assertIn('overall_grade', analysis)
        self.assertIn('overall_grade_percentage', analysis)
        self.assertIn('decisions', analysis)
        self.assertIn('key_insights', analysis)
        self.assertIn('learning_points', analysis)
        
        # Verify grade is valid
        self.assertIn(analysis['overall_grade'], ['A', 'B', 'C', 'D', 'F'])
        self.assertGreaterEqual(analysis['overall_grade_percentage'], 0)
        self.assertLessEqual(analysis['overall_grade_percentage'], 100)
    
    def test_hand_analysis_with_multiple_actions(self):
        """Test hand analysis with multiple actions across streets"""
        # Start game
        self.app.post('/api/game/start',
                     json={'session_id': self.session_id})
        
        # Create comprehensive hand history
        hand_history = [
            {'player_id': 0, 'action': 1, 'stage': 0, 'pot': 4, 'hand': [('S', 'A'), ('H', 'K')], 'public_cards': [], 'stakes': [100, 100]},
            {'player_id': 0, 'action': 1, 'stage': 1, 'pot': 8, 'hand': [('S', 'A'), ('H', 'K')], 'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10')], 'stakes': [100, 100]},
            {'player_id': 0, 'action': 1, 'stage': 2, 'pot': 16, 'hand': [('S', 'A'), ('H', 'K')], 'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10'), ('C', '9')], 'stakes': [100, 100]}
        ]
        
        game_state = {
            'pot': 16,
            'stage': 2,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10'), ('C', '9')],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        self.assertEqual(response.status_code, 200)
        analysis = json.loads(response.data)
        
        # Verify multiple decisions analyzed
        self.assertGreater(len(analysis.get('decisions', [])), 0)
        self.assertIn('overall_grade', analysis)


class TestChatInterfaceE2E(unittest.TestCase):
    """Test chat interface flow end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_e2e_chat'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_complete_chat_workflow(self, mock_openai):
        """Test complete workflow: send message → receive response with context"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test response from the AI coach."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Step 1: Send initial message
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'What is GTO strategy?'
                                })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertIn('timestamp', data)
        self.assertGreater(len(data['response']), 0)
        
        # Step 2: Send follow-up message (should have context)
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'Can you explain more?'
                                })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_with_game_context(self, mock_openai):
        """Test chat with game context injection"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Based on your current hand..."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Start game
        self.app.post('/api/game/start',
                     json={'session_id': self.session_id})
        
        # Get game state
        response = self.app.get(f'/api/game/state?session_id={self.session_id}')
        game_state = json.loads(response.data)
        
        # Send chat with game context
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'What should I do?',
                                    'game_context': game_state
                                })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)


class TestErrorHandlingE2E(unittest.TestCase):
    """Test error handling end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_e2e_errors'
    
    def test_api_failure_handling(self):
        """Test handling of API failures"""
        # Test with invalid endpoint
        response = self.app.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        # Test with missing required fields
        response = self.app.post('/api/coach/analyze-hand', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_network_error_simulation(self):
        """Test handling of network errors (simulated)"""
        # Test with invalid JSON
        response = self.app.post('/api/coach/chat',
                                data='invalid json',
                                content_type='application/json')
        # Allow 400 or 500 for invalid JSON
        self.assertIn(response.status_code, [400, 500])
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        # Test with wrong data types
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': 123,  # Should be string
                                    'hand_history': 'not a list',
                                    'game_state': 'not a dict'
                                })
        self.assertEqual(response.status_code, 400)
        
        # Test with empty required fields
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': '',
                                    'message': ''
                                })
        self.assertEqual(response.status_code, 400)


class TestEdgeCasesE2E(unittest.TestCase):
    """Test edge cases end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_e2e_edge_cases'
    
    def test_all_in_scenario_e2e(self):
        """Test all-in scenario end-to-end"""
        # Start game
        self.app.post('/api/game/start',
                     json={'session_id': self.session_id})
        
        # Create all-in hand history
        hand_history = [
            {
                'player_id': 0,
                'action': 4,  # All-in
                'stage': 0,
                'pot': 200,
                'hand': [('S', 'A'), ('H', 'A')],
                'public_cards': [],
                'stakes': [100, 100]  # All-in
            }
        ]
        
        game_state = {
            'pot': 200,
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'A')],
            'public_cards': [],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        self.assertEqual(response.status_code, 200)
        analysis = json.loads(response.data)
        self.assertIn('overall_grade', analysis)
    
    def test_first_hand_no_history_e2e(self):
        """Test first hand with no history end-to-end"""
        # Start game
        self.app.post('/api/game/start',
                     json={'session_id': self.session_id})
        
        # Analyze with empty history
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': [],
                                    'game_state': {'pot': 4, 'stage': 0}
                                })
        self.assertEqual(response.status_code, 200)
        analysis = json.loads(response.data)
        self.assertIn('overall_grade', analysis)
        # Should handle gracefully
        self.assertEqual(len(analysis.get('decisions', [])), 0)
    
    def test_very_long_hand_e2e(self):
        """Test very long hand end-to-end"""
        # Create very long hand history
        long_history = []
        for stage in range(4):
            for i in range(10):
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
            'pot': 500,
            'stage': 3,
            'hand': [('S', 'A'), ('H', 'K')],
            'public_cards': [('S', 'Q'), ('H', 'J'), ('D', '10'), ('C', '9')],
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': long_history,
                                    'game_state': game_state
                                })
        self.assertEqual(response.status_code, 200)
        analysis = json.loads(response.data)
        self.assertIn('overall_grade', analysis)
        # Should handle long hands without crashing
        self.assertGreater(len(analysis.get('decisions', [])), 0)


if __name__ == '__main__':
    unittest.main()

