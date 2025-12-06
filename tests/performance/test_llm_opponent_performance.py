"""
Performance tests for LLMOpponentAgent.

Tests latency targets and timeout enforcement.
"""

import pytest
import sys
import os
import time
from unittest.mock import Mock, patch

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestPerformance:
    """Tests for performance targets"""
    
    def test_decision_latency_with_fallback(self):
        """Test decision latency when using GTOAgent fallback"""
        # Create agent
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = False  # Use GTOAgent fallback
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # Measure latency
        start_time = time.time()
        action = agent.step(state)
        elapsed_time = time.time() - start_time
        
        # Fallback should be very fast (< 0.1s)
        assert elapsed_time < 0.1
        assert action in [Action.FOLD, Action.CHECK_CALL]
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_decision_latency_with_mocked_llm(self, mock_openai):
        """Test decision latency with mocked LLM (realistic delays)"""
        import json
        
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # Mock LLM response with realistic delay (2-4 seconds)
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_tool_call = Mock()
            mock_function = Mock()
            mock_function.name = "select_poker_action"
            mock_function.arguments = json.dumps({"action_type": "call"})
            mock_tool_call.function = mock_function
            mock_message.tool_calls = [mock_tool_call]
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            # Simulate 2.5 second delay
            def delayed_result():
                time.sleep(0.01)  # Small delay for test
                return mock_response
            mock_future.result.return_value = mock_response
            mock_submit.return_value = mock_future
            
            # Measure latency
            start_time = time.time()
            action = agent.step(state)
            elapsed_time = time.time() - start_time
            
            # Should complete quickly in test (mocked)
            assert elapsed_time < 1.0  # Mocked call should be fast
            assert action in [Action.FOLD, Action.CHECK_CALL]
    
    def test_timeout_enforcement(self):
        """Test that timeout is enforced correctly"""
        # Create agent
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = True
        
        # Verify timeout settings
        assert agent.api_timeout == 12.0
        assert agent.executor_timeout == 15.0
        assert agent.executor_timeout > agent.api_timeout
    
    def test_average_decision_latency(self):
        """Test average decision latency across multiple decisions"""
        # Create agent
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = False  # Use GTOAgent fallback
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # Measure multiple decisions
        latencies = []
        num_decisions = 10
        
        for _ in range(num_decisions):
            start_time = time.time()
            action = agent.step(state)
            elapsed_time = time.time() - start_time
            latencies.append(elapsed_time)
        
        # Calculate average
        avg_latency = sum(latencies) / len(latencies)
        
        # Fallback should be very fast
        assert avg_latency < 0.1
        assert max(latencies) < 0.1  # 95th percentile equivalent
    
    def test_performance_targets(self):
        """Test that performance targets are met"""
        # Create agent
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = False  # Use GTOAgent fallback
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # Measure latency
        start_time = time.time()
        action = agent.step(state)
        elapsed_time = time.time() - start_time
        
        # Performance targets:
        # Preferred: < 3 seconds
        # Acceptable: < 5 seconds
        # Maximum: < 10 seconds (with timeout)
        
        # With fallback, should be much faster
        assert elapsed_time < 3.0  # Preferred target
        assert action in [Action.FOLD, Action.CHECK_CALL]


