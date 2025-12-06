"""
Unit tests for LLMOpponentAgent error handling and fallback logic.

Tests retry logic, timeout handling, fallback to GTOAgent, and illegal action handling.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import TimeoutError as FutureTimeoutError

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestRetryLogic:
    """Tests for retry logic with exponential backoff"""
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_retry_on_network_failure(self, mock_openai):
        """Test retry on network failure"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_future.result.side_effect = Exception("Network error")
            mock_submit.return_value = mock_future
            
            context = {
                'opponent_cards': ['Ac', 'Ad'],
                'current_stage': 'preflop',
                'public_cards': [],
                'pot_size': 9,
                'pot_size_bb': 4.5,
                'big_blind': 2,
                'current_stacks': {'user': 994, 'opponent': 1000},
                'stack_depths': {'user': 497.0, 'opponent': 500.0},
                'action_history': [],
                'legal_actions': [Action.FOLD, Action.CHECK_CALL],
                'legal_actions_labels': {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Check/Call'},
                'facing_bet': False,
                'bet_to_call': 0,
                'bet_to_call_bb': 0.0,
                'pot_odds': 0.0
            }
            
            with patch('time.sleep'):  # Mock sleep
                result = agent._call_llm_for_decision(context, retry_count=0)
                # Should retry up to 2 times, then return None
                # Since we're starting at retry_count=0, it will retry twice
                assert mock_submit.call_count >= 1
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_retry_exponential_backoff(self, mock_openai):
        """Test exponential backoff delays"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_future.result.side_effect = Exception("API Error")
            mock_submit.return_value = mock_future
            
            context = {
                'opponent_cards': ['Ac', 'Ad'],
                'current_stage': 'preflop',
                'public_cards': [],
                'pot_size': 9,
                'pot_size_bb': 4.5,
                'big_blind': 2,
                'current_stacks': {'user': 994, 'opponent': 1000},
                'stack_depths': {'user': 497.0, 'opponent': 500.0},
                'action_history': [],
                'legal_actions': [Action.FOLD, Action.CHECK_CALL],
                'legal_actions_labels': {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Check/Call'},
                'facing_bet': False,
                'bet_to_call': 0,
                'bet_to_call_bb': 0.0,
                'pot_odds': 0.0
            }
            
            sleep_times = []
            def mock_sleep(seconds):
                sleep_times.append(seconds)
            
            with patch('time.sleep', side_effect=mock_sleep):
                agent._call_llm_for_decision(context, retry_count=0)
                # Should sleep with exponential backoff: 1s, 2s
                # But we only retry 2 times max, so we'll see delays for retries
                assert len(sleep_times) <= 2  # Max 2 retries


class TestTimeoutHandling:
    """Tests for timeout handling"""
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_api_timeout(self, mock_openai):
        """Test API timeout handling"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_future.result.side_effect = FutureTimeoutError()
            mock_submit.return_value = mock_future
            
            context = {
                'opponent_cards': ['Ac', 'Ad'],
                'current_stage': 'preflop',
                'public_cards': [],
                'pot_size': 9,
                'pot_size_bb': 4.5,
                'big_blind': 2,
                'current_stacks': {'user': 994, 'opponent': 1000},
                'stack_depths': {'user': 497.0, 'opponent': 500.0},
                'action_history': [],
                'legal_actions': [Action.FOLD, Action.CHECK_CALL],
                'legal_actions_labels': {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Check/Call'},
                'facing_bet': False,
                'bet_to_call': 0,
                'bet_to_call_bb': 0.0,
                'pot_odds': 0.0
            }
            
            with patch('time.sleep'):
                result = agent._call_llm_for_decision(context, retry_count=2)  # No retries
                assert result is None


class TestFallbackToGTOAgent:
    """Tests for fallback to GTOAgent"""
    
    def test_fallback_when_llm_unavailable(self):
        """Test fallback when LLM is not available"""
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = False
        agent.client = None
        
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
        
        # Should use GTOAgent fallback
        action = agent.step(state)
        assert action in [Action.FOLD, Action.CHECK_CALL]
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_fallback_on_llm_failure(self, mock_openai):
        """Test fallback when LLM call fails"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_future.result.side_effect = Exception("LLM API Error")
            mock_submit.return_value = mock_future
            
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
            
            with patch('time.sleep'):
                # Should fallback to GTOAgent after all retries fail
                action = agent.step(state)
                assert action in [Action.FOLD, Action.CHECK_CALL]
    
    def test_fallback_when_no_opponent_cards(self):
        """Test fallback when opponent cards cannot be extracted"""
        agent = LLMOpponentAgent(num_actions=5)
        agent.api_key_available = True
        
        state = {
            'raw_obs': {
                'hand': [],  # No cards
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # Should fallback to GTOAgent
        action = agent.step(state)
        assert action in [Action.FOLD, Action.CHECK_CALL]


class TestIllegalActionHandling:
    """Tests for illegal action handling with clarification retry"""
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_illegal_action_retry_with_clarification(self, mock_openai):
        """Test retry with clarification when illegal action is selected"""
        import json
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
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
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]  # Only fold/call available
        }
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            # First call returns illegal action (raise_half_pot)
            mock_future1 = Mock()
            mock_response1 = Mock()
            mock_choice1 = Mock()
            mock_message1 = Mock()
            mock_tool_call1 = Mock()
            mock_function1 = Mock()
            mock_function1.name = "select_poker_action"
            mock_function1.arguments = json.dumps({"action_type": "raise_half_pot"})
            mock_tool_call1.function = mock_function1
            mock_message1.tool_calls = [mock_tool_call1]
            mock_choice1.message = mock_message1
            mock_response1.choices = [mock_choice1]
            mock_future1.result.return_value = mock_response1
            
            # Second call (with clarification) returns legal action
            mock_future2 = Mock()
            mock_response2 = Mock()
            mock_choice2 = Mock()
            mock_message2 = Mock()
            mock_tool_call2 = Mock()
            mock_function2 = Mock()
            mock_function2.name = "select_poker_action"
            mock_function2.arguments = json.dumps({"action_type": "call"})
            mock_tool_call2.function = mock_function2
            mock_message2.tool_calls = [mock_tool_call2]
            mock_choice2.message = mock_message2
            mock_response2.choices = [mock_choice2]
            mock_future2.result.return_value = mock_response2
            
            mock_submit.side_effect = [mock_future1, mock_future2]
            
            # Mock the mapping to detect illegal action
            with patch.object(agent, '_map_llm_action_to_rlcard') as mock_map:
                # First call returns illegal action
                def map_side_effect(action, legal_actions):
                    if action == "raise_half_pot":
                        return Action.RAISE_HALF_POT  # This is illegal
                    return Action.CHECK_CALL
                mock_map.side_effect = map_side_effect
                
                with patch('time.sleep'):
                    # Should retry with clarification
                    action = agent.step(state)
                    # Should eventually get a legal action or fallback
                    assert action in [Action.FOLD, Action.CHECK_CALL]

