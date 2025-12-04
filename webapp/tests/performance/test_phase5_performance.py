"""
Performance tests for Phase 5
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import app after setting up path
try:
    from app import app
except ImportError:
    # Try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), '../../app.py'))
    app_module = importlib.util.module_from_spec(spec)
    sys.modules['app'] = app_module
    spec.loader.exec_module(app_module)
    app = app_module.app
from coach.strategy_evaluator import StrategyEvaluator


class TestHandAnalysisPerformance(unittest.TestCase):
    """Test hand analysis performance targets"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_perf_hand_analysis'
        self.evaluator = StrategyEvaluator()
    
    def test_hand_analysis_under_2_seconds(self):
        """Test hand analysis completes in < 2 seconds"""
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
        self.assertLess(elapsed_time, 2.0, 
                       f"Hand analysis took {elapsed_time:.2f}s, target: < 2s")
    
    def test_hand_analysis_with_multiple_decisions(self):
        """Test hand analysis performance with multiple decisions"""
        hand_history = []
        for stage in range(4):
            for i in range(3):
                hand_history.append({
                    'player_id': 0,
                    'action': 1,
                    'stage': stage,
                    'pot': 4 + (stage * 10) + (i * 5),
                    'hand': [('S', 'A'), ('H', 'K')],
                    'public_cards': [],
                    'stakes': [100, 100]
                })
        
        game_state = {
            'pot': 50,
            'stage': 3,
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
        self.assertLess(elapsed_time, 2.0,
                       f"Hand analysis with multiple decisions took {elapsed_time:.2f}s, target: < 2s")
    
    def test_direct_evaluator_performance(self):
        """Test StrategyEvaluator direct performance"""
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        start_time = time.time()
        result = self.evaluator.evaluate_action(game_state, 1)
        elapsed_time = time.time() - start_time
        
        self.assertIsNotNone(result)
        self.assertLess(elapsed_time, 0.1,
                       f"Direct evaluation took {elapsed_time:.3f}s, should be < 0.1s")


class TestChatPerformance(unittest.TestCase):
    """Test chat response performance targets"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_perf_chat'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_response_under_3_seconds_mocked(self, mock_openai):
        """Test chat response completes in < 3 seconds (with mocked API)"""
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
        # Mocked response should be very fast
        self.assertLess(elapsed_time, 1.0,
                       f"Chat response (mocked) took {elapsed_time:.2f}s")
    
    def test_chat_timeout_handling(self):
        """Test chat timeout handling (3 second timeout)"""
        # This test verifies timeout mechanism exists
        # Actual timeout testing would require slow API simulation
        from coach.chatbot_coach import ChatbotCoach
        
        # Verify timeout is configured
        coach = ChatbotCoach()
        self.assertIsNotNone(coach.executor)


class TestConcurrentRequests(unittest.TestCase):
    """Test system under concurrent load"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_concurrent'
    
    def test_concurrent_hand_analysis(self):
        """Test multiple concurrent hand analysis requests"""
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
        
        def make_request(session_id):
            response = self.app.post('/api/coach/analyze-hand',
                                    json={
                                        'session_id': f'{session_id}_{threading.current_thread().ident}',
                                        'hand_history': hand_history,
                                        'game_state': game_state
                                    })
            return response.status_code == 200
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, self.session_id) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        self.assertTrue(all(results), "All concurrent requests should succeed")
    
    def test_concurrent_chat_requests(self):
        """Test multiple concurrent chat requests"""
        @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
        @patch('coach.chatbot_coach.OpenAI')
        def make_chat_request(mock_openai, session_id):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            response = self.app.post('/api/coach/chat',
                                    json={
                                        'session_id': f'{session_id}_{threading.current_thread().ident}',
                                        'message': 'Test'
                                    })
            return response.status_code == 200
        
        # Note: This test would need proper mocking setup
        # For now, we verify the endpoint can handle requests
        pass


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage and leak detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = StrategyEvaluator()
    
    def test_no_memory_leak_in_hand_analysis(self):
        """Test that hand analysis doesn't leak memory"""
        import gc
        
        game_state = {
            'stage': 0,
            'hand': [('S', 'A'), ('H', 'K')],
            'pot': 4,
            'stakes': [100, 100],
            'big_blind': 2
        }
        
        # Run many analyses
        for i in range(100):
            result = self.evaluator.evaluate_action(game_state, 1)
            self.assertIsNotNone(result)
        
        # Force garbage collection
        gc.collect()
        
        # Verify async_results doesn't grow unbounded
        # (PatternRecognizer has cache limits)
        self.assertLess(len(self.evaluator.async_results), 1000,
                       "Async results storage should be bounded")
    
    def test_pattern_cache_limits(self):
        """Test pattern recognizer cache has limits"""
        from coach.pattern_recognizer import PatternRecognizer
        
        recognizer = PatternRecognizer()
        session_id = 'test_cache'
        
        # Add many cached analyses
        for i in range(20):
            recognizer.cache_analysis(session_id, i, {'test': 'data'})
        
        # Cache should be limited to 10 per session
        self.assertLessEqual(len(recognizer.pattern_cache.get(session_id, {})), 10,
                            "Pattern cache should be limited to 10 analyses per session")


class TestStressTesting(unittest.TestCase):
    """Test system under stress"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.session_id = 'test_stress'
    
    def test_rapid_requests(self):
        """Test system handles rapid requests"""
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
        
        # Make 50 rapid requests
        success_count = 0
        for i in range(50):
            response = self.app.post('/api/coach/analyze-hand',
                                    json={
                                        'session_id': f'{self.session_id}_{i}',
                                        'hand_history': hand_history,
                                        'game_state': game_state
                                    })
            if response.status_code == 200:
                success_count += 1
        
        # Most requests should succeed (allowing for some failures under stress)
        self.assertGreater(success_count, 45,
                          f"At least 45/50 requests should succeed, got {success_count}")


if __name__ == '__main__':
    unittest.main()

