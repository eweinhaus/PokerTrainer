"""
Pot calculation utility for accurate pot size calculation in heads-up no-limit poker.

This module provides a single source of truth for pot calculation, handling:
- Cumulative pot calculation (across all betting rounds using in_chips)
- Current-round pot calculation (for individual betting rounds using raised)
- Preflop pot calculation (blinds + raises)
- Postflop pot calculation (matched bets)
- Unit conversion (chips to BB)
"""

import logging

logger = logging.getLogger(__name__)



def calculate_cumulative_pot(in_chips):
    """
    Calculate cumulative pot from player in_chips (total chips invested).

    Args:
        in_chips: List of chips invested by each player

    Returns:
        Total pot size (sum of all invested chips)
    """
    return sum(int(chips) for chips in in_chips)


def calculate_pot(raised, big_blind, dealer_id=None, stage=0):
    """
    Calculate pot accurately for any betting round (CURRENT ROUND ONLY).

    IMPORTANT: This calculates pot for the CURRENT betting round only.
    For cumulative pot across all rounds, use calculate_cumulative_pot() with in_chips.

    In heads-up no-limit poker:
    - Preflop: Pot = SB + BB + all raises
    - Postflop: Pot = sum of all matched bets (raised array should be equal for both players)

    Args:
        raised: List of bets in current betting round [player0_bet, player1_bet] in chips
        big_blind: Big blind amount in chips
        dealer_id: Dealer position (BB in HUNL). If None, assumes raised includes blinds.
        stage: Current betting stage (0=preflop, 1=flop, 2=turn, 3=river)

    Returns:
        Pot size for current betting round in chips (integer)
    """
    if not raised or len(raised) < 2:
        return 0
    
    # Convert to integers to avoid numpy type issues
    raised = [int(r) if r else 0 for r in raised]
    big_blind = int(big_blind) if big_blind else 2
    
    # For postflop, pot is simply the sum of matched bets
    # (raised array should be equal for both players after betting round ends)
    if stage > 0:
        # Postflop: pot = sum of all bets (should be matched)
        pot = sum(raised)
        return pot
    
    # Preflop: need to ensure blinds are included
    small_blind = big_blind // 2
    
    if dealer_id is None:
        # If dealer_id not provided, assume raised array already includes blinds
        # This is a fallback for cases where dealer_id is not available
        pot = sum(raised)
        return pot
    else:
        # Dealer_id provided, determine SB/BB positions
        bb_pos = dealer_id  # Dealer is BB in HUNL
        sb_pos = (dealer_id + 1) % 2  # Other player is SB
        
        # Check if blinds are already included in raised
        sb_in_raised = raised[sb_pos] >= small_blind
        bb_in_raised = raised[bb_pos] >= big_blind
        
        if sb_in_raised and bb_in_raised:
            # Blinds already included, just sum raised
            pot = sum(raised)
        else:
            # Add missing blinds
            effective_sb = max(raised[sb_pos], small_blind)
            effective_bb = max(raised[bb_pos], big_blind)
            pot = effective_sb + effective_bb
        
        return pot

def calculate_cumulative_pot(in_chips):
    """
    Calculate cumulative pot from player in_chips (total chips invested).

    Args:
        in_chips: List of chips invested by each player

    Returns:
        Total pot size (sum of all invested chips)
    """
    if not in_chips:
        return 0
    return sum(int(chips) if chips is not None else 0 for chips in in_chips)


def calculate_pot_from_state(raw_obs, env=None):
    """
    Calculate cumulative pot from game state.

    For preflop: Uses raised array (reconstructed from action history) for current betting round pot.
    For postflop: Uses in_chips for cumulative pot across all betting rounds.

    Args:
        raw_obs: Raw observation dictionary from RLCard
        env: RLCard environment (optional, for dealer_id)

    Returns:
        Cumulative pot size in chips (integer)
    """
    # Get stage
    stage = raw_obs.get('stage', 0)
    if hasattr(stage, 'value'):
        stage = stage.value
    elif not isinstance(stage, int):
        stage = int(stage) if stage else 0
    
    # For postflop, use raised array which should now be cumulative (reconstructed from action history)
    if stage > 0:
        raised = raw_obs.get('raised', [0, 0])
        raised_sum = sum(int(r) if r else 0 for r in raised)

        if raised_sum > 0:
            # Use the cumulative raised array to calculate pot
            big_blind = raw_obs.get('big_blind', 2)
            dealer_id = None
            if env is not None:
                if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                    dealer_id = env.game.dealer_id
                elif hasattr(env, 'dealer_id'):
                    dealer_id = env.dealer_id
            pot = calculate_pot(raised, big_blind, dealer_id, 0)  # Use stage 0 for cumulative calculation
            logger.info(f"💰 [POT_CALC_POSTFLOP] Using cumulative raised array: {raised} -> pot: {pot} chips")
            return pot

        # Fallback: Try to get from env.game.players (most reliable for cumulative pot)
        if env is not None and hasattr(env, 'game') and hasattr(env.game, 'players'):
            try:
                in_chips = []
                for p in env.game.players:
                    chip_value = p.in_chips
                    # Handle numpy types
                    if hasattr(chip_value, 'item'):
                        chip_value = chip_value.item()
                    in_chips.append(int(chip_value))

                if len(in_chips) >= 2:
                    pot = calculate_cumulative_pot(in_chips)
                    logger.info(f"💰 [POT_CALC_POSTFLOP] Using in_chips from env.game.players: {in_chips} -> pot: {pot} chips")
                    return pot
            except Exception as e:
                logger.warning(f"💰 [POT_CALC_POSTFLOP] Failed to get in_chips from env.game.players: {e}")
                import traceback
                logger.warning(f"💰 [POT_CALC_POSTFLOP] Traceback: {traceback.format_exc()}")
                pass

        # Secondary: Calculate from stakes (remaining chips)
        stakes = raw_obs.get('stakes', None)
        if stakes and len(stakes) >= 2:
            try:
                # For heads-up, assume standard starting stack of 100
                starting_stack = 100
                invested = [starting_stack - int(s) for s in stakes]
                pot = sum(invested)
                logger.info(f"💰 [POT_CALC_POSTFLOP] Using stakes: {stakes}, invested: {invested} -> pot: {pot} chips")
                return pot
            except Exception as e:
                logger.warning(f"💰 [POT_CALC_POSTFLOP] Failed to calculate pot from stakes: {e}")
                pass

        # If we get here, something went wrong - log and use fallback
        logger.warning(f"💰 [POT_CALC_POSTFLOP] All methods failed, using raised array fallback. Stage: {stage}, raw_obs keys: {list(raw_obs.keys())}")
        # Fallback: Use raised array for current betting round only (not cumulative)
        raised = raw_obs.get('raised', [0, 0])
        big_blind = raw_obs.get('big_blind', 2)
        dealer_id = None
        if env is not None:
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id
            elif hasattr(env, 'dealer_id'):
                dealer_id = env.dealer_id
        pot = calculate_pot(raised, big_blind, dealer_id, stage)
        logger.warning(f"💰 [POT_CALC_POSTFLOP] Using raised array fallback: {raised} -> pot: {pot} chips (NOTE: This is current round only, not cumulative)")
        return pot
    
    # For preflop: Use raised array calculation (reconstructed from action history)
    raised = raw_obs.get('raised', [0, 0])
    raised_sum = sum(int(r) if r else 0 for r in raised)
    
    if raised_sum > 0:  # If raised array has values, use it (most accurate for preflop)
        big_blind = raw_obs.get('big_blind', 2)
        
        # Get dealer_id from env if available
        dealer_id = None
        if env is not None:
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id
            elif hasattr(env, 'dealer_id'):
                dealer_id = env.dealer_id
        
        pot = calculate_pot(raised, big_blind, dealer_id, stage)
        return pot

    # Fallback: Try to get from env.game.players (for preflop if raised array is empty)
    if env is not None and hasattr(env, 'game') and hasattr(env.game, 'players'):
        try:
            in_chips = [int(p.in_chips) for p in env.game.players]
            if len(in_chips) >= 2:
                pot = calculate_cumulative_pot(in_chips)
                return pot
        except Exception as e:
            logger.warning(f"Failed to get in_chips from env.game.players: {e}")
            pass

    # Last resort: Use raised array even if zero (for edge cases)
    big_blind = raw_obs.get('big_blind', 2)
    dealer_id = None
    if env is not None:
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
        elif hasattr(env, 'dealer_id'):
            dealer_id = env.dealer_id
    
    pot = calculate_pot(raised, big_blind, dealer_id, stage)
    return pot


def pot_to_bb(pot_chips, big_blind):
    """
    Convert pot from chips to big blinds.
    
    Args:
        pot_chips: Pot size in chips
        big_blind: Big blind amount in chips
    
    Returns:
        Pot size in BB (float)
    """
    if big_blind == 0:
        return 0.0
    
    return float(pot_chips) / float(big_blind)


def get_display_pot(pot_chips, raised, big_blind, include_unmatched=True):
    """
    Get pot for display purposes.
    
    During a betting round, there may be unmatched bets (one player has bet more than the other).
    This function can optionally exclude unmatched bets for display.
    
    Args:
        pot_chips: Total pot in chips (may include unmatched bets)
        raised: Current round bet amounts [player0, player1]
        big_blind: Big blind amount
        include_unmatched: Whether to include unmatched bets
    
    Returns:
        Display pot in BB
    """
    if include_unmatched:
        return pot_to_bb(pot_chips, big_blind)
    
    # Exclude unmatched bets
    if len(raised) == 2:
        matched_amount = min(raised[0], raised[1])
        matched_pot = matched_amount * 2  # Both players contribute matched amount
        return pot_to_bb(matched_pot, big_blind)
    
    return pot_to_bb(pot_chips, big_blind)

    # For preflop, add blinds if not already included in raised
    if stage == 0:  # Preflop
        # Determine SB and BB positions
        if dealer_id is not None:
            # In HUNL: dealer is BB, so SB is (dealer_id + 1) % 2
            bb_pos = dealer_id  # Dealer is BB
            sb_pos = (dealer_id + 1) % 2  # Other player is SB
        else:
            # Fallback: assume player 0 is SB, player 1 is BB
            sb_pos = 0
            bb_pos = 1
        
        # Check if blinds are already included in raised array
        # If raised[sb_pos] < SB amount, blinds are not included
        sb_amount = big_blind // 2
        bb_amount = big_blind
        
        blinds_included = (raised[sb_pos] >= sb_amount and raised[bb_pos] >= bb_amount)
        
        if not blinds_included:
            # Add blinds to the calculation
            effective_sb = max(raised[sb_pos], sb_amount)
            effective_bb = max(raised[bb_pos], bb_amount)
            return effective_sb + effective_bb
        
        # Blinds already included, just sum raised
        return sum(raised)
    
    else:  # Postflop
        # For postflop, raised should represent matched bets
        # But in case of unmatched bets, take the minimum
        return min(raised) * 2  # Both players contribute the matched amount
