"""
Basic tests for LLMOpponentAgent

Tests core functionality without requiring LLM API calls.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestLLMOpponentAgentBasic:
    """Basic tests for LLMOpponentAgent"""
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly"""
        agent = LLMOpponentAgent(num_actions=5)
        assert agent.use_raw == True
        assert agent.num_actions == 5
        assert hasattr(agent, 'gto_agent')  # Fallback agent should be initialized
    
    def test_card_conversion(self):
        """Test card index to rank/suit conversion"""
        agent = LLMOpponentAgent(num_actions=5)
        
        # Test card 0 (Ace of clubs)
        rank, suit = agent._card_to_rank_suit(0)
        assert rank == 0  # Ace
        assert suit == 0  # Clubs
        
        # Test card 51 (King of spades)
        rank, suit = agent._card_to_rank_suit(51)
        assert rank == 12  # King
        assert suit == 3  # Spades
    
    def test_hand_to_string(self):
        """Test hand to string conversion"""
        agent = LLMOpponentAgent(num_actions=5)
        
        # Test pair
        hand_str = agent._hand_to_string([0, 13])  # Ace of clubs, Ace of diamonds
        assert hand_str == "AA"
        
        # Test suited
        hand_str = agent._hand_to_string([0, 1])  # Ace of clubs, 2 of clubs
        assert hand_str == "A2s"
        
        # Test offsuit
        hand_str = agent._hand_to_string([0, 14])  # Ace of clubs, 2 of diamonds
        assert hand_str == "A2o"
    
    def test_card_index_to_string(self):
        """Test card index to string conversion"""
        agent = LLMOpponentAgent(num_actions=5)
        
        # Test Ace of clubs
        card_str = agent._card_index_to_string(0)
        assert card_str == "Ac"
        
        # Test King of spades
        card_str = agent._card_index_to_string(51)
        assert card_str == "Ks"
    
    def test_extract_opponent_cards(self):
        """Test opponent card extraction"""
        agent = LLMOpponentAgent(num_actions=5)
        
        # Test valid state
        state = {
            'raw_obs': {
                'hand': [0, 13]  # Two aces
            }
        }
        cards = agent._extract_opponent_cards(state)
        assert cards == [0, 13]
        
        # Test invalid state
        state = {
            'raw_obs': {
                'hand': []
            }
        }
        cards = agent._extract_opponent_cards(state)
        assert cards is None
    
    def test_action_mapping(self):
        """Test LLM action to RLCard action mapping"""
        agent = LLMOpponentAgent(num_actions=5)
        
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        
        # Test all mappings
        assert agent._map_llm_action_to_rlcard("fold", legal_actions) == Action.FOLD
        assert agent._map_llm_action_to_rlcard("call", legal_actions) == Action.CHECK_CALL
        assert agent._map_llm_action_to_rlcard("check", legal_actions) == Action.CHECK_CALL
        assert agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions) == Action.RAISE_HALF_POT
        assert agent._map_llm_action_to_rlcard("raise_pot", legal_actions) == Action.RAISE_POT
        assert agent._map_llm_action_to_rlcard("all_in", legal_actions) == Action.ALL_IN
    
    def test_action_mapping_fallback(self):
        """Test action mapping fallback logic"""
        agent = LLMOpponentAgent(num_actions=5)
        
        # Test fallback when raise_half_pot is illegal but raise_pot is legal
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT]
        action = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        assert action == Action.RAISE_POT
        
        # Test fallback when raise_pot is illegal but raise_half_pot is legal
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        action = agent._map_llm_action_to_rlcard("raise_pot", legal_actions)
        assert action == Action.RAISE_HALF_POT
    
    def test_build_context(self):
        """Test context building"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],  # Opponent cards
                'public_cards': [26, 27, 28],  # Board cards
                'pot': 100,
                'big_blind': 2,
                'all_chips': [1000, 1000],
                'raised': [0, 0],
                'stage': 1  # Flop
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        }
        
        context = agent._build_context(state)
        
        # Verify required fields
        assert 'opponent_cards' in context
        assert 'current_stage' in context
        assert 'public_cards' in context
        assert 'pot_size' in context
        assert 'legal_actions' in context
        assert 'action_history' in context
        assert context['current_stage'] == 'flop'
        assert len(context['opponent_cards']) == 2
        assert len(context['public_cards']) == 3
    
    def test_format_context_for_prompt(self):
        """Test context formatting"""
        agent = LLMOpponentAgent(num_actions=5)
        
        context = {
            'opponent_cards': ['Ac', 'Ad'],
            'current_stage': 'preflop',
            'public_cards': [],
            'pot_size': 100,
            'pot_size_bb': 50.0,
            'big_blind': 2,
            'current_stacks': {'user': 1000, 'opponent': 1000},
            'stack_depths': {'user': 500.0, 'opponent': 500.0},
            'action_history': [],
            'legal_actions': [Action.FOLD, Action.CHECK_CALL],
            'legal_actions_labels': {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Check/Call'},
            'facing_bet': False,
            'bet_to_call': 0,
            'bet_to_call_bb': 0.0,
            'pot_odds': 0.0
        }
        
        formatted = agent._format_context_for_prompt(context)
        
        # Verify formatted text contains key information
        assert 'Current Game Situation' in formatted
        assert 'Ac' in formatted
        assert 'preflop' in formatted.lower()
        assert 'Legal Actions Available' in formatted
        assert 'Fold' in formatted
    
    def test_system_prompt(self):
        """Test system prompt building"""
        agent = LLMOpponentAgent(num_actions=5)
        
        prompt = agent._build_system_prompt()
        
        # Verify prompt contains key GTO principles
        assert 'GTO' in prompt
        assert 'Range-Based Thinking' in prompt
        assert 'Pot Odds' in prompt
        assert 'Position' in prompt
        assert 'select_poker_action' in prompt
    
    def test_tool_calling_schema(self):
        """Test tool calling schema"""
        agent = LLMOpponentAgent(num_actions=5)
        
        schema = agent._get_tool_calling_schema()
        
        # Verify schema structure
        assert len(schema) == 1
        assert schema[0]['type'] == 'function'
        assert schema[0]['function']['name'] == 'select_poker_action'
        assert 'action_type' in schema[0]['function']['parameters']['properties']
        assert 'reasoning' in schema[0]['function']['parameters']['properties']
    
    def test_eval_step(self):
        """Test eval_step method"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 100,
                'big_blind': 2,
                'all_chips': [1000, 1000],
                'raised': [0, 0],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL]
        }
        
        # This will use GTOAgent fallback if LLM is not available
        action, info = agent.eval_step(state)
        
        # Verify return format
        assert action in [Action.FOLD, Action.CHECK_CALL]
        assert 'probs' in info
        assert isinstance(info['probs'], dict)

