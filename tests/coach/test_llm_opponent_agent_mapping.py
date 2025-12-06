"""
Unit tests for LLMOpponentAgent action mapping and validation logic.

Tests action mapping from LLM tool calls to RLCard Action enum.
"""

import pytest
import sys
import os

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestActionMapping:
    """Tests for _map_llm_action_to_rlcard()"""
    
    def test_map_fold(self):
        """Test mapping fold action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        result = agent._map_llm_action_to_rlcard("fold", legal_actions)
        assert result == Action.FOLD
    
    def test_map_call(self):
        """Test mapping call action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        result = agent._map_llm_action_to_rlcard("call", legal_actions)
        assert result == Action.CHECK_CALL
    
    def test_map_check(self):
        """Test mapping check action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        result = agent._map_llm_action_to_rlcard("check", legal_actions)
        assert result == Action.CHECK_CALL
    
    def test_map_raise_half_pot(self):
        """Test mapping raise_half_pot action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        
        result = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        assert result == Action.RAISE_HALF_POT
    
    def test_map_raise_pot(self):
        """Test mapping raise_pot action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT]
        
        result = agent._map_llm_action_to_rlcard("raise_pot", legal_actions)
        assert result == Action.RAISE_POT
    
    def test_map_all_in(self):
        """Test mapping all_in action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.ALL_IN]
        
        result = agent._map_llm_action_to_rlcard("all_in", legal_actions)
        assert result == Action.ALL_IN


class TestActionMappingFallback:
    """Tests for fallback logic in action mapping"""
    
    def test_fallback_raise_half_pot_to_raise_pot(self):
        """Test fallback when raise_half_pot is illegal but raise_pot is legal"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT]
        
        result = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        assert result == Action.RAISE_POT
    
    def test_fallback_raise_pot_to_raise_half_pot(self):
        """Test fallback when raise_pot is illegal but raise_half_pot is legal"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        
        result = agent._map_llm_action_to_rlcard("raise_pot", legal_actions)
        assert result == Action.RAISE_HALF_POT
    
    def test_fallback_call_to_check_call(self):
        """Test fallback when call is illegal but CHECK_CALL is legal (shouldn't happen)"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        result = agent._map_llm_action_to_rlcard("call", legal_actions)
        assert result == Action.CHECK_CALL
    
    def test_fallback_last_resort(self):
        """Test last resort fallback to first legal action"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        # Try to map an action that's completely illegal
        result = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        # Should fall back to first legal action
        assert result in legal_actions
        assert result == legal_actions[0]  # First legal action


class TestIllegalActionHandling:
    """Tests for illegal action detection and handling"""
    
    def test_illegal_action_detection(self):
        """Test that illegal actions are detected"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        # Try to map an illegal action
        result = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        # Should fall back to first legal action
        assert result in legal_actions
    
    def test_illegal_action_with_fallback_options(self):
        """Test illegal action with available fallback options"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT]
        
        # Try raise_half_pot which is illegal, but raise_pot is available
        result = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        assert result == Action.RAISE_POT  # Should use fallback
    
    def test_all_actions_illegal(self):
        """Test handling when all raise actions are illegal"""
        agent = LLMOpponentAgent(num_actions=5)
        legal_actions = [Action.FOLD, Action.CHECK_CALL]
        
        # Try various illegal actions
        result1 = agent._map_llm_action_to_rlcard("raise_half_pot", legal_actions)
        result2 = agent._map_llm_action_to_rlcard("raise_pot", legal_actions)
        result3 = agent._map_llm_action_to_rlcard("all_in", legal_actions)
        
        # All should fall back to first legal action
        assert result1 in legal_actions
        assert result2 in legal_actions
        assert result3 in legal_actions


