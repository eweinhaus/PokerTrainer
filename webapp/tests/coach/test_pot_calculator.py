"""
Unit tests for pot calculation utility.

Tests cover:
- Preflop pot calculation (blinds + raises)
- Postflop pot calculation (matched bets)
- Edge cases and error handling
- Unit conversion (chips to BB)
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from coach.pot_calculator import (
    calculate_pot,
    calculate_cumulative_pot,
    calculate_pot_from_state,
    pot_to_bb,
    get_display_pot
)


class TestPotCalculator(unittest.TestCase):
    """Test cases for pot calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.big_blind = 2  # 2 chips = 1 BB
        self.small_blind = 1  # 1 chip = 0.5 BB
        self.dealer_id = 1  # In HUNL, dealer = BB
    
    def test_preflop_blinds_only(self):
        """Test pot calculation with only blinds posted."""
        # Scenario: Button folds immediately
        # SB posts 1 chip, BB posts 2 chips
        raised = [0, 2]  # Button folded (0), BB posted (2)
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: SB (1) + BB (2) = 3 chips = 1.5 BB
        self.assertEqual(pot, 3)
    
    def test_preflop_button_calls(self):
        """Test pot calculation when button calls (completes to BB)."""
        # Scenario: Button calls, BB checks
        # SB posts 1 chip, completes to 2 chips (adds 1 more)
        # BB posts 2 chips
        raised = [2, 2]  # Both matched at 2 chips
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 2 + 2 = 4 chips = 2 BB
        self.assertEqual(pot, 4)
    
    def test_preflop_button_opens_to_3bb(self):
        """Test pot calculation when button opens to 3 BB."""
        # Scenario: Button raises to 3 BB total
        # SB posts 1 chip, raises to 6 chips total (adds 5 more)
        # BB posts 2 chips
        raised = [6, 2]  # Button 6 chips (3 BB), BB 2 chips (1 BB)
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 6 + 2 = 8 chips = 4 BB
        self.assertEqual(pot, 8)
    
    def test_preflop_button_opens_bb_calls(self):
        """Test pot calculation when button opens and BB calls."""
        # Scenario: Button opens to 3 BB, BB calls
        # Both players have 6 chips in (3 BB each)
        raised = [6, 6]  # Both matched at 6 chips
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 6 + 6 = 12 chips = 6 BB
        self.assertEqual(pot, 12)
    
    def test_preflop_button_opens_bb_3bets(self):
        """Test pot calculation when button opens and BB 3-bets."""
        # Scenario: Button opens to 3 BB, BB 3-bets to 10 BB
        # Button: 6 chips (3 BB)
        # BB: 20 chips (10 BB)
        raised = [6, 20]  # Button 6 chips, BB 20 chips
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 6 + 20 = 26 chips = 13 BB
        self.assertEqual(pot, 26)
    
    def test_preflop_button_opens_bb_3bets_button_4bets(self):
        """Test pot calculation when button opens, BB 3-bets, button 4-bets."""
        # Scenario: Button opens to 3 BB, BB 3-bets to 10 BB, Button 4-bets to 25 BB
        # Button: 50 chips (25 BB)
        # BB: 20 chips (10 BB)
        raised = [50, 20]  # Button 50 chips, BB 20 chips
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 50 + 20 = 70 chips = 35 BB
        self.assertEqual(pot, 70)
    
    def test_preflop_raised_includes_blinds(self):
        """Test pot calculation when raised array already includes blinds."""
        # Scenario: Raised array includes blinds
        # SB: 1 chip (already in raised)
        # BB: 2 chips (already in raised)
        raised = [1, 2]  # Blinds already included
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: 1 + 2 = 3 chips = 1.5 BB
        self.assertEqual(pot, 3)

    def test_calculate_cumulative_pot(self):
        """Test cumulative pot calculation across all betting rounds."""
        # Test various scenarios
        in_chips = [6, 6]  # Both players 6 chips each
        pot = calculate_cumulative_pot(in_chips)
        self.assertEqual(pot, 12)  # 6 + 6 = 12 chips

        in_chips = [14, 14]  # Both players 14 chips each
        pot = calculate_cumulative_pot(in_chips)
        self.assertEqual(pot, 28)  # 14 + 14 = 28 chips

        in_chips = [30, 30]  # Both players 30 chips each
        pot = calculate_cumulative_pot(in_chips)
        self.assertEqual(pot, 60)  # 30 + 30 = 60 chips

        # Test edge cases
        in_chips = [0, 0]  # No bets
        pot = calculate_cumulative_pot(in_chips)
        self.assertEqual(pot, 0)

        in_chips = [10, 5]  # Unequal bets
        pot = calculate_cumulative_pot(in_chips)
        self.assertEqual(pot, 15)  # 10 + 5 = 15 chips

    def test_calculate_cumulative_pot_invalid_input(self):
        """Test cumulative pot calculation with invalid inputs."""
        # Empty list
        pot = calculate_cumulative_pot([])
        self.assertEqual(pot, 0)

        # Single element
        pot = calculate_cumulative_pot([10])
        self.assertEqual(pot, 10)

        # None values
        pot = calculate_cumulative_pot([None, None])
        self.assertEqual(pot, 0)

    def test_calculate_pot_from_state_cumulative(self):
        """Test calculate_pot_from_state with cumulative pot (in_chips)."""
        # Mock env with players that have in_chips
        class MockPlayer:
            def __init__(self, in_chips):
                self.in_chips = in_chips

        class MockGame:
            def __init__(self, in_chips_list):
                self.players = [MockPlayer(ic) for ic in in_chips_list]

        class MockEnv:
            def __init__(self, in_chips_list):
                self.game = MockGame(in_chips_list)

        # Test preflop + flop scenario
        raw_obs = {
            'raised': [14, 14],  # Cumulative bets (reconstructed)
            'big_blind': 2,
            'stage': 1,  # Flop
            'in_chips': [14, 14]  # Cumulative for hand
        }

        env = MockEnv([14, 14])  # in_chips = [14, 14]
        pot = calculate_pot_from_state(raw_obs, env)
        # Should use cumulative raised array: 14 + 14 = 28 chips
        self.assertEqual(pot, 28)

    def test_calculate_pot_from_state_fallback(self):
        """Test calculate_pot_from_state fallback to current-round calculation."""
        # Mock env without in_chips
        class MockEnv:
            pass

        raw_obs = {
            'raised': [8, 8],  # Current round bets
            'big_blind': 2,
            'stage': 1  # Flop
        }

        env = MockEnv()
        pot = calculate_pot_from_state(raw_obs, env)
        # Should fall back to current-round pot: 8 + 8 = 16 chips
        self.assertEqual(pot, 16)

    def test_full_hand_pot_progression(self):
        """Test pot progression across a full hand."""
        # Mock env
        class MockPlayer:
            def __init__(self, in_chips):
                self.in_chips = in_chips

        class MockGame:
            def __init__(self, in_chips_list):
                self.players = [MockPlayer(ic) for ic in in_chips_list]

        class MockEnv:
            def __init__(self, in_chips_list):
                self.game = MockGame(in_chips_list)

        # Scenario: Button opens to 3 BB, BB calls, then BB bets flop, Button calls
        # Preflop: in_chips = [6, 6] (both 3 BB)
        raw_obs_preflop = {'raised': [6, 6], 'big_blind': 2, 'stage': 0}
        env_preflop = MockEnv([6, 6])
        pot_preflop = calculate_pot_from_state(raw_obs_preflop, env_preflop)
        self.assertEqual(pot_preflop, 12)  # 6 + 6 = 12 chips

        # Flop: cumulative raised = [14, 14] (preflop 6 + flop 8 each)
        raw_obs_flop = {'raised': [14, 14], 'big_blind': 2, 'stage': 1}
        env_flop = MockEnv([14, 14])
        pot_flop = calculate_pot_from_state(raw_obs_flop, env_flop)
        self.assertEqual(pot_flop, 28)  # 14 + 14 = 28 chips

        # Turn: Both check, cumulative raised unchanged = [14, 14]
        raw_obs_turn = {'raised': [14, 14], 'big_blind': 2, 'stage': 2}
        env_turn = MockEnv([14, 14])
        pot_turn = calculate_pot_from_state(raw_obs_turn, env_turn)
        self.assertEqual(pot_turn, 28)  # Still 14 + 14 = 28 chips

        # River: cumulative raised = [30, 30] (preflop 6 + flop 8 + turn 0 + river 16 each)
        raw_obs_river = {'raised': [30, 30], 'big_blind': 2, 'stage': 3}
        env_river = MockEnv([30, 30])
        pot_river = calculate_pot_from_state(raw_obs_river, env_river)
        self.assertEqual(pot_river, 60)  # 30 + 30 = 60 chips
    
    def test_preflop_raised_excludes_blinds(self):
        """Test pot calculation when raised array excludes blinds."""
        # Scenario: Raised array doesn't include blinds (edge case)
        # This shouldn't happen in normal play, but test for robustness
        raised = [0, 0]  # No bets beyond blinds
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=0)
        # Pot should be: SB (1) + BB (2) = 3 chips = 1.5 BB
        self.assertEqual(pot, 3)
    
    def test_preflop_no_dealer_id(self):
        """Test pot calculation when dealer_id is not provided."""
        # Scenario: Dealer ID not available (fallback case)
        raised = [6, 2]  # Button 6 chips, BB 2 chips
        pot = calculate_pot(raised, self.big_blind, dealer_id=None, stage=0)
        # Should still calculate correctly (assumes raised includes blinds)
        # Pot should be: 6 + 2 = 8 chips = 4 BB
        self.assertEqual(pot, 8)
    
    def test_postflop_matched_bets(self):
        """Test pot calculation for postflop with matched bets."""
        # Scenario: Both players checked postflop
        # Postflop: raised array should be equal for both players
        raised = [0, 0]  # Both checked (no new bets)
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=1)
        # Pot should be: 0 + 0 = 0 chips (no new bets this round)
        # Note: Actual pot includes preflop bets, but this is just for this betting round
        self.assertEqual(pot, 0)
    
    def test_postflop_bb_bets_sb_calls(self):
        """Test pot calculation for postflop when BB bets and SB calls."""
        # Scenario: BB bets 4 BB, SB calls
        # BB: 8 chips (4 BB)
        # SB: 8 chips (4 BB)
        raised = [8, 8]  # Both matched at 8 chips
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=1)
        # Pot should be: 8 + 8 = 16 chips = 8 BB
        self.assertEqual(pot, 16)
    
    def test_postflop_unmatched_bets(self):
        """Test pot calculation for postflop with unmatched bets."""
        # Scenario: BB bets 4 BB, SB hasn't acted yet
        # BB: 8 chips (4 BB)
        # SB: 0 chips (hasn't acted)
        raised = [0, 8]  # Unmatched bets
        pot = calculate_pot(raised, self.big_blind, self.dealer_id, stage=1)
        # Pot should be: 0 + 8 = 8 chips = 4 BB
        # (Unmatched bets are still part of pot during betting round)
        self.assertEqual(pot, 8)
    
    def test_pot_to_bb_conversion(self):
        """Test conversion from chips to big blinds."""
        # Test various pot sizes
        self.assertEqual(pot_to_bb(3, 2), 1.5)  # 3 chips = 1.5 BB
        self.assertEqual(pot_to_bb(4, 2), 2.0)  # 4 chips = 2 BB
        self.assertEqual(pot_to_bb(8, 2), 4.0)  # 8 chips = 4 BB
        self.assertEqual(pot_to_bb(12, 2), 6.0)  # 12 chips = 6 BB
        self.assertEqual(pot_to_bb(26, 2), 13.0)  # 26 chips = 13 BB
    
    def test_pot_to_bb_zero_big_blind(self):
        """Test conversion with zero big blind (edge case)."""
        # Should return 0 and log warning
        result = pot_to_bb(10, 0)
        self.assertEqual(result, 0.0)
    
    def test_get_display_pot_include_unmatched(self):
        """Test display pot calculation including unmatched bets."""
        pot_chips = 8  # Total pot
        raised = [0, 8]  # Unmatched bets
        big_blind = 2
        
        # Include unmatched bets
        display_pot = get_display_pot(pot_chips, raised, big_blind, include_unmatched=True)
        self.assertEqual(display_pot, 4.0)  # 8 chips / 2 BB = 4 BB
    
    def test_get_display_pot_exclude_unmatched(self):
        """Test display pot calculation excluding unmatched bets."""
        pot_chips = 8  # Total pot (includes unmatched)
        raised = [0, 8]  # Unmatched bets
        big_blind = 2
        
        # Exclude unmatched bets
        display_pot = get_display_pot(pot_chips, raised, big_blind, include_unmatched=False)
        # Matched pot = 0 (both players matched at 0)
        # Unmatched = 8, so matched_pot_from_total = 8 - 8 = 0
        self.assertEqual(display_pot, 0.0)
    
    def test_calculate_pot_from_state_preflop(self):
        """Test calculate_pot_from_state for preflop scenario."""
        raw_obs = {
            'raised': [6, 2],  # Button 6 chips, BB 2 chips
            'big_blind': 2,
            'stage': 0  # Preflop
        }
        
        # Mock env with dealer_id
        class MockEnv:
            class MockGame:
                dealer_id = 1
            game = MockGame()
        
        env = MockEnv()
        pot = calculate_pot_from_state(raw_obs, env)
        # Pot should be: 6 + 2 = 8 chips = 4 BB
        self.assertEqual(pot, 8)
    
    def test_calculate_pot_from_state_postflop(self):
        """Test calculate_pot_from_state for postflop scenario."""
        raw_obs = {
            'raised': [8, 8],  # Both matched at 8 chips
            'big_blind': 2,
            'stage': 1  # Flop
        }
        
        # Mock env with dealer_id
        class MockEnv:
            class MockGame:
                dealer_id = 1
            game = MockGame()
        
        env = MockEnv()
        pot = calculate_pot_from_state(raw_obs, env)
        # Pot should be: 8 + 8 = 16 chips = 8 BB
        self.assertEqual(pot, 16)
    
    def test_calculate_pot_from_state_no_env(self):
        """Test calculate_pot_from_state without env (fallback)."""
        raw_obs = {
            'raised': [6, 2],  # Button 6 chips, BB 2 chips
            'big_blind': 2,
            'stage': 0  # Preflop
        }
        
        pot = calculate_pot_from_state(raw_obs, env=None)
        # Should still calculate (uses fallback logic)
        # Pot should be: 6 + 2 = 8 chips = 4 BB
        self.assertEqual(pot, 8)
    
    def test_calculate_pot_invalid_raised(self):
        """Test pot calculation with invalid raised array."""
        # Empty raised array
        pot = calculate_pot([], self.big_blind, self.dealer_id, stage=0)
        self.assertEqual(pot, 0)
        
        # Single element
        pot = calculate_pot([6], self.big_blind, self.dealer_id, stage=0)
        self.assertEqual(pot, 0)
    
    def test_calculate_pot_numpy_types(self):
        """Test pot calculation with numpy types (common in RLCard)."""
        import numpy as np
        
        # Test with numpy integers
        raised = [np.int64(6), np.int64(2)]
        big_blind = np.int64(2)
        pot = calculate_pot(raised, big_blind, self.dealer_id, stage=0)
        self.assertEqual(pot, 8)
    
    def test_calculate_pot_large_blinds(self):
        """Test pot calculation with larger blind sizes."""
        # Test with BB = 4 chips
        big_blind = 4
        small_blind = 2
        raised = [12, 4]  # Button 12 chips (3 BB), BB 4 chips (1 BB)
        pot = calculate_pot(raised, big_blind, self.dealer_id, stage=0)
        # Pot should be: 12 + 4 = 16 chips = 4 BB
        self.assertEqual(pot, 16)
    
    def test_calculate_pot_different_dealer(self):
        """Test pot calculation with different dealer position."""
        # Dealer = 0 (SB in HUNL)
        dealer_id = 0
        # In HUNL: dealer = BB, so BB is player 0, SB is player 1
        raised = [2, 6]  # BB (player 0) 2 chips, SB (player 1) 6 chips
        pot = calculate_pot(raised, self.big_blind, dealer_id, stage=0)
        # Pot should be: 2 + 6 = 8 chips = 4 BB
        self.assertEqual(pot, 8)


if __name__ == '__main__':
    unittest.main()

