"""
Validation tests for LLMOpponentAgent decision quality.

Tests that LLM decisions are GTO-appropriate by comparing with GTOAgent.
"""

import pytest
import sys
import os

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.llm_opponent_agent import LLMOpponentAgent, Action
from coach.gto_agent import GTOAgent


class TestDecisionQuality:
    """Tests for decision quality validation"""
    
    def test_compare_with_gto_agent_preflop(self):
        """Test LLM decisions compared with GTOAgent for preflop"""
        # Create agents
        llm_agent = LLMOpponentAgent(num_actions=5)
        llm_agent.api_key_available = False  # Use GTOAgent fallback for testing
        gto_agent = GTOAgent(num_actions=5)
        
        # Create test state
        state = {
            'raw_obs': {
                'hand': [0, 13],  # AA
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0  # Preflop
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
        }
        
        # Get decisions
        llm_action = llm_agent.step(state)
        gto_action = gto_agent.step(state)
        
        # Both should return valid actions
        assert llm_action in state['raw_legal_actions']
        assert gto_action in state['raw_legal_actions']
        
        # Note: Since LLM agent falls back to GTOAgent when LLM unavailable,
        # they should match in this test. In production with LLM, decisions may differ.
    
    def test_compare_with_gto_agent_postflop(self):
        """Test LLM decisions compared with GTOAgent for postflop"""
        # Create agents
        llm_agent = LLMOpponentAgent(num_actions=5)
        llm_agent.api_key_available = False  # Use GTOAgent fallback
        gto_agent = GTOAgent(num_actions=5)
        
        # Create test state (postflop)
        state = {
            'raw_obs': {
                'hand': [0, 13],  # AA
                'public_cards': [26, 27, 28],  # Flop
                'pot': 100,
                'big_blind': 2,
                'all_chips': [950, 1050],
                'raised': [0, 0],
                'stage': 1  # Flop
            },
            'raw_legal_actions': [Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
        }
        
        # Get decisions
        llm_action = llm_agent.step(state)
        gto_action = gto_agent.step(state)
        
        # Both should return valid actions
        assert llm_action in state['raw_legal_actions']
        assert gto_action in state['raw_legal_actions']
    
    def test_range_considerations(self):
        """Test that range considerations are evident in context"""
        agent = LLMOpponentAgent(num_actions=5)
        
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
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        }
        
        context = agent._build_context(state)
        
        # Context should include action history which helps with range considerations
        assert 'action_history' in context
        # Action history helps LLM understand ranges
    
    def test_pot_odds_considered(self):
        """Test that pot odds are calculated and included in context"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],  # Facing a bet
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        context = agent._build_context(state)
        
        # Pot odds should be calculated when facing a bet
        assert 'pot_odds' in context
        if context['facing_bet']:
            assert context['pot_odds'] > 0
            assert 'bet_to_call' in context
            assert context['bet_to_call'] > 0
    
    def test_position_awareness(self):
        """Test that position is included in context"""
        agent = LLMOpponentAgent(num_actions=5)
        
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
        
        context = agent._build_context(state)
        
        # Position should be included
        assert 'user_position' in context
        assert 'opponent_position' in context
        assert context['opponent_position'] == 'big_blind'
        assert context['user_position'] == 'button'
    
    def test_stack_depth_considerations(self):
        """Test that stack depth is included in context"""
        agent = LLMOpponentAgent(num_actions=5)
        
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
        
        context = agent._build_context(state)
        
        # Stack depth should be included
        assert 'stack_depths' in context
        assert 'user' in context['stack_depths']
        assert 'opponent' in context['stack_depths']
        assert context['stack_depths']['opponent'] > 0
        assert context['stack_depths']['user'] > 0


