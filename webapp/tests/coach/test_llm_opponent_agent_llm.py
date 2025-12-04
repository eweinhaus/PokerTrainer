"""
Unit tests for LLMOpponentAgent LLM integration with mocked API calls.

Tests LLM API calls, tool calling responses, retry logic, and timeout handling.
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import TimeoutError as FutureTimeoutError

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestLLMIntegration:
    """Tests for LLM API integration with mocked calls"""
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_successful_llm_call(self, mock_openai):
        """Test successful LLM call with tool calling"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Create mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_tool_call = Mock()
        mock_function = Mock()
        
        mock_function.name = "select_poker_action"
        mock_function.arguments = json.dumps({"action_type": "call", "reasoning": "Good pot odds"})
        mock_tool_call.function = mock_function
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Mock the executor submit
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_future.result.return_value = mock_response
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
            
            result = agent._call_llm_for_decision(context)
            
            assert result == "call"
            mock_submit.assert_called_once()
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_llm_call_timeout(self, mock_openai):
        """Test LLM call timeout handling"""
        # Setup mock client
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
            
            # Should return None after timeout (no retries in test to keep it fast)
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = agent._call_llm_for_decision(context, retry_count=2)  # Start at max retries
                assert result is None
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_llm_call_retry_on_failure(self, mock_openai):
        """Test retry logic on LLM call failure"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            # First call fails, second succeeds
            mock_future1 = Mock()
            mock_future1.result.side_effect = Exception("API Error")
            mock_future2 = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_tool_call = Mock()
            mock_function = Mock()
            mock_function.name = "select_poker_action"
            mock_function.arguments = json.dumps({"action_type": "fold"})
            mock_tool_call.function = mock_function
            mock_message.tool_calls = [mock_tool_call]
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_future2.result.return_value = mock_response
            
            mock_submit.side_effect = [mock_future1, mock_future2]
            
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
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = agent._call_llm_for_decision(context, retry_count=0)
                assert result == "fold"
                assert mock_submit.call_count == 2  # Should retry once
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_llm_call_invalid_response(self, mock_openai):
        """Test handling of invalid LLM response"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_response = Mock()
            mock_response.choices = []  # No choices
            mock_future.result.return_value = mock_response
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
            
            result = agent._call_llm_for_decision(context, retry_count=2)  # No retries
            assert result is None
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_llm_call_no_tool_calls(self, mock_openai):
        """Test handling of response without tool calls"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.tool_calls = None  # No tool calls
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_future.result.return_value = mock_response
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
            
            result = agent._call_llm_for_decision(context, retry_count=2)  # No retries
            assert result is None


class TestToolCallParsing:
    """Tests for parsing tool call responses"""
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_parse_tool_call_arguments(self, mock_openai):
        """Test parsing of tool call arguments"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = LLMOpponentAgent(num_actions=5)
        agent.client = mock_client
        agent.api_key_available = True
        
        with patch.object(agent.executor, 'submit') as mock_submit:
            mock_future = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_tool_call = Mock()
            mock_function = Mock()
            
            # Test with reasoning
            mock_function.name = "select_poker_action"
            mock_function.arguments = json.dumps({
                "action_type": "raise_half_pot",
                "reasoning": "Strong hand, good equity"
            })
            mock_tool_call.function = mock_function
            mock_message.tool_calls = [mock_tool_call]
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_future.result.return_value = mock_response
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
                'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT],
                'legal_actions_labels': {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Check/Call', Action.RAISE_HALF_POT: 'Raise ½ Pot'},
                'facing_bet': False,
                'bet_to_call': 0,
                'bet_to_call_bb': 0.0,
                'pot_odds': 0.0
            }
            
            result = agent._call_llm_for_decision(context)
            assert result == "raise_half_pot"

