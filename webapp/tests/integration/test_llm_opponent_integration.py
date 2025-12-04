"""
Integration tests for LLMOpponentAgent complete decision flow.

Tests complete flow from state to action using real RLCard game states.
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import rlcard
from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestCompleteDecisionFlow:
    """Tests for complete decision flow from state to action"""
    
    def test_complete_flow_preflop(self):
        """Test complete decision flow for preflop"""
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        agent.api_key_available = False  # Use GTOAgent fallback for testing
        
        # Reset environment
        state, player_id = env.reset()
        
        # If it's not opponent's turn, advance to opponent's turn
        if player_id != 1:
            # Make a dummy action for player 0
            state, player_id = env.step(Action.CHECK_CALL)
        
        # Get state for player 1 (opponent)
        if player_id == 1:
            state = env.get_state(player_id)
            
            # Test complete flow
            action = agent.step(state)
            
            # Verify action is valid
            assert action in [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
    
    def test_complete_flow_postflop(self):
        """Test complete decision flow for postflop"""
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        agent.api_key_available = False  # Use GTOAgent fallback for testing
        
        # Reset environment
        state, player_id = env.reset()
        
        # Advance to postflop by playing through preflop
        action_count = 0
        max_actions = 20
        
        while not env.is_over() and action_count < max_actions:
            # Get current state to check stage
            temp_state = env.get_state(0)
            raw_obs = temp_state.get('raw_obs', {})
            stage = raw_obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage = stage.value
            
            if stage > 0:  # Postflop reached
                break
            
            # Get current player from state
            state, player_id = env.get_state(0), 0
            if player_id == 0:
                # Player 0 action
                state, player_id = env.step(Action.CHECK_CALL)
            else:
                # Opponent action
                state = env.get_state(1)
                action = agent.step(state)
                state, player_id = env.step(action)
            
            action_count += 1
        
        # If we reached postflop, test opponent decision
        if not env.is_over():
            # Check if it's opponent's turn
            state, player_id = env.get_state(0), 0
            if player_id == 1:  # Opponent's turn
                state = env.get_state(1)
                action = agent.step(state)
                
                # Verify action is valid
                assert action in [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
    
    @patch('coach.llm_opponent_agent.OpenAI')
    def test_complete_flow_with_mocked_llm(self, mock_openai):
        """Test complete decision flow with mocked LLM"""
        import json
        
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        agent.client = mock_client
        agent.api_key_available = True
        
        # Reset environment
        state, player_id = env.reset()
        
        # Advance to opponent's turn
        if player_id != 1:
            state, player_id = env.step(Action.CHECK_CALL)
        
        if player_id == 1:
            state = env.get_state(player_id)
            
            # Mock LLM response
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
                mock_future.result.return_value = mock_response
                mock_submit.return_value = mock_future
                
                # Test complete flow
                action = agent.step(state)
                
                # Verify action is valid
                assert action in [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
                # Should be call based on mocked response
                assert action == Action.CHECK_CALL
    
    def test_context_building_accuracy(self):
        """Test that context building is accurate"""
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        
        # Reset environment
        state, player_id = env.reset()
        
        # Advance to opponent's turn
        if player_id != 1:
            state, player_id = env.step(Action.CHECK_CALL)
        
        if player_id == 1:
            state = env.get_state(player_id)
            
            # Build context
            context = agent._build_context(state)
            
            # Verify context accuracy
            assert 'opponent_cards' in context
            assert len(context['opponent_cards']) == 2
            assert 'current_stage' in context
            assert 'legal_actions' in context
            assert len(context['legal_actions']) > 0
            assert 'pot_size' in context
            assert context['pot_size'] > 0
    
    def test_action_mapping_and_validation(self):
        """Test action mapping and validation"""
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        
        # Reset environment
        state, player_id = env.reset()
        
        # Advance to opponent's turn
        if player_id != 1:
            state, player_id = env.step(Action.CHECK_CALL)
        
        if player_id == 1:
            state = env.get_state(player_id)
            raw_legal_actions = state.get('raw_legal_actions', [])
            
            # Test mapping various actions
            test_actions = ["fold", "call", "check", "raise_half_pot", "raise_pot", "all_in"]
            
            for llm_action in test_actions:
                mapped_action = agent._map_llm_action_to_rlcard(llm_action, raw_legal_actions)
                # Mapped action should be in legal actions (or fallback to first legal)
                assert mapped_action in raw_legal_actions or len(raw_legal_actions) == 0
    
    def test_error_handling_and_fallback(self):
        """Test error handling and fallback to GTOAgent"""
        # Create RLCard environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agent
        agent = LLMOpponentAgent(num_actions=env.num_actions)
        agent.api_key_available = False  # Force fallback
        
        # Reset environment
        state, player_id = env.reset()
        
        # Advance to opponent's turn
        if player_id != 1:
            state, player_id = env.step(Action.CHECK_CALL)
        
        if player_id == 1:
            state = env.get_state(player_id)
            
            # Should use GTOAgent fallback
            action = agent.step(state)
            
            # Verify action is valid
            assert action in [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]

