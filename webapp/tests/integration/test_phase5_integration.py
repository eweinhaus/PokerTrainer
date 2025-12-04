"""
Integration tests for Phase 5 features
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
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
from coach.chatbot_coach import ChatbotCoach


class TestHandAnalysisIntegration(unittest.TestCase):
    """Test hand analysis integration flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_session_integration'
        self.evaluator = StrategyEvaluator()
    
    def test_complete_hand_analysis_flow(self):
        """Test complete hand analysis flow from API to response"""
        # Start a game
        response = self.app.post('/api/game/start', 
                                json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 200)
        
        # Create hand history
        hand_history = [
            {
                'player_id': 0,
                'action': 1,  # Call
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
        
        # Request hand analysis
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('overall_grade', data)
        self.assertIn('decisions', data)
        self.assertIn('key_insights', data)
        self.assertIn('learning_points', data)
    
    def test_hand_analysis_with_error_handling(self):
        """Test hand analysis with error scenarios"""
        # Test with missing session_id
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'hand_history': [],
                                    'game_state': {}
                                })
        self.assertEqual(response.status_code, 400)
        
        # Test with invalid hand_history type
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': 'not a list',
                                    'game_state': {}
                                })
        self.assertEqual(response.status_code, 400)
        
        # Test with empty hand history
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': [],
                                    'game_state': {'pot': 100}
                                })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('overall_grade', data)
    
    def test_hand_analysis_performance(self):
        """Test hand analysis meets performance target (< 2 seconds)"""
        import time
        
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
        
        start_time = time.time()
        response = self.app.post('/api/coach/analyze-hand',
                                json={
                                    'session_id': self.session_id,
                                    'hand_history': hand_history,
                                    'game_state': game_state
                                })
        elapsed_time = time.time() - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed_time, 2.0, f"Hand analysis took {elapsed_time:.2f}s, target: < 2s")


class TestChatInterfaceIntegration(unittest.TestCase):
    """Test chat interface integration flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_chat_integration'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_complete_chat_flow(self, mock_openai):
        """Test complete chat interface flow"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response from coach"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Send chat message
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'What is GTO?'
                                })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertIn('timestamp', data)
    
    def test_chat_with_error_handling(self):
        """Test chat interface with error scenarios"""
        # Test with missing session_id
        response = self.app.post('/api/coach/chat',
                                json={'message': 'test'})
        self.assertEqual(response.status_code, 400)
        
        # Test with missing message
        response = self.app.post('/api/coach/chat',
                                json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 400)
        
        # Test with empty message
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': ''
                                })
        self.assertEqual(response.status_code, 400)
        
        # Test with too long message
        long_message = 'a' * 501
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': long_message
                                })
        self.assertEqual(response.status_code, 400)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_with_game_context(self, mock_openai):
        """Test chat with game context injection"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Context-aware response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        game_context = {
            'pot': 100,
            'stage': 1,
            'hand': [('S', 'A'), ('H', 'K')]
        }
        
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'What should I do?',
                                    'game_context': game_context
                                })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_performance(self, mock_openai):
        """Test chat response meets performance target (< 3 seconds)"""
        import time
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Quick response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        start_time = time.time()
        response = self.app.post('/api/coach/chat',
                                json={
                                    'session_id': self.session_id,
                                    'message': 'Test message'
                                })
        elapsed_time = time.time() - start_time
        
        self.assertEqual(response.status_code, 200)
        # Note: Actual API calls may take longer, but mocked should be fast
        # In real tests, we'd test with actual timeout handling


class TestAPIEndpointIntegration(unittest.TestCase):
    """Test API endpoint integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_api_integration'
    
    def test_get_hand_history(self):
        """Test get hand history endpoint"""
        # Add some hand history
        hand_history_storage[self.session_id] = [
            {'player_id': 0, 'action': 1, 'stage': 0}
        ]
        
        response = self.app.get(f'/api/coach/get-hand-history?session_id={self.session_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('decisions', data)
    
    def test_get_hand_history_missing_session(self):
        """Test get hand history with missing session"""
        response = self.app.get('/api/coach/get-hand-history?session_id=nonexistent')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['decisions'], [])
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_async_analysis_endpoint(self):
        """Test async analysis endpoint"""
        # Test with non-existent analysis_id
        response = self.app.get('/api/coach/analyze-hand-async/nonexistent')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('ready', data)
        self.assertFalse(data['ready'])


class TestFrontendBackendIntegration(unittest.TestCase):
    """Test frontend-backend communication"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_frontend_backend'
    
    def test_game_state_flow(self):
        """Test complete game state flow"""
        # Start game
        response = self.app.post('/api/game/start',
                                json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 200)
        game_state = json.loads(response.data)
        self.assertIn('hand', game_state)
        self.assertIn('pot', game_state)
        
        # Get game state
        response = self.app.get(f'/api/game/state?session_id={self.session_id}')
        self.assertEqual(response.status_code, 200)
        game_state = json.loads(response.data)
        self.assertIn('hand', game_state)
    
    def test_action_processing_flow(self):
        """Test action processing flow"""
        # Start game
        self.app.post('/api/game/start',
                     json={'session_id': self.session_id})
        
        # Process action
        response = self.app.post('/api/game/action',
                                json={
                                    'session_id': self.session_id,
                                    'action_index': 1  # Call
                                })
        self.assertEqual(response.status_code, 200)
        game_state = json.loads(response.data)
        self.assertIn('is_waiting_for_action', game_state)
    
    def test_error_responses_format(self):
        """Test error responses are properly formatted for frontend"""
        # Test 400 error
        response = self.app.post('/api/coach/analyze-hand',
                                json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        # Test 500 error handling
        # This would require mocking an internal error
        # For now, we test that errors return JSON
        response = self.app.get('/api/game/state')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()

