"""
GTO Rules and Ranges

This module contains Game Theory Optimal (GTO) preflop ranges and utility functions
for No Limit Hold'em strategy evaluation.

Ranges are based on published HUNL optimal ranges from established sources.
"""


class GTORules:
    """
    GTO rules and ranges for No Limit Hold'em evaluation.
    
    Preflop ranges are based on published HUNL optimal ranges (100 BB stack depth).
    Sources: Upswing Poker, PokerSnowie, and other established GTO resources.
    """
    
    def __init__(self):
        """Initialize GTO rules with preflop ranges."""
        self._initialize_preflop_ranges()
    
    def _initialize_preflop_ranges(self):
        """Initialize preflop range lookup tables for all stack depths and scenarios."""
        # Base ranges for 100+ BB (standard stack depth)
        self._initialize_base_ranges()
        
        # Stack depth specific ranges
        self._initialize_stack_depth_ranges()
        
        # All-in ranges
        self._initialize_allin_ranges()
    
    def _initialize_base_ranges(self):
        """Initialize base preflop ranges for 100+ BB stack depth."""
        # Button opening range (~70-75% of hands for 100+ BB)
        self.button_opening_range_100bb = {
            # Premium hands - always open
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            
            # Strong hands - always open
            'TT': 'raise', '99': 'raise', '88': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'KQs': 'raise', 'KQo': 'raise',
            
            # Medium hands - open most of the time
            '77': 'raise', '66': 'raise', '55': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'KJs': 'raise', 'KJo': 'raise',
            'QJs': 'raise', 'QJo': 'raise',
            'JTs': 'raise', 'JTo': 'raise',
            
            # Weak hands - open selectively (mixed strategies)
            '44': 'raise', '33': 'raise', '22': 'raise',
            'A9s': 'raise', 'A8s': 'raise', 'A7s': 'raise', 'A6s': 'raise', 'A5s': 'raise',
            'A4s': 'raise', 'A3s': 'raise', 'A2s': 'raise',
            'KTs': 'raise', 'K9s': 'raise',
            'QTs': 'raise', 'Q9s': 'raise',
            'J9s': 'raise', 'T9s': 'raise', '98s': 'raise', '87s': 'raise',
            '76s': 'raise', '65s': 'raise', '54s': 'raise', '43s': 'raise', '42s': 'raise', '32s': 'raise',
            # Mixed strategy hands - raise with frequency, fold otherwise
            'A9o': 0.7, 'A8o': 0.7, 'A7o': 0.65, 'A6o': 0.65, 'A5o': 0.7,  # A5o is better due to wheel potential
            'A4o': 0.6, 'A3o': 0.6, 'A2o': 0.6,
            'KTo': 0.75, 'K9o': 0.7,
            'QTo': 0.7, 'Q9o': 0.65,
            'JTo': 0.7, 'J9o': 0.65,
            'T9o': 0.65, 'T8o': 0.6, 'T7o': 0.55,
            '98o': 0.7, '97o': 0.6, '96o': 0.55,
            '87o': 0.65, '86o': 0.6, '85o': 0.55,
            '76o': 0.7, '75o': 0.6, '74o': 0.55,
            '65o': 0.7, '64o': 0.6, '63o': 0.55,
            '54o': 0.7, '53o': 0.6, '52o': 0.55,
            '43o': 0.6, '42o': 0.5, '32o': 0.5,
            # Trash hands - always fold
            'K8o': 'fold', 'K7o': 'fold', 'K6o': 'fold', 'K5o': 'fold', 'K4o': 'fold', 'K3o': 'fold', 'K2o': 'fold',
            'Q8o': 'fold', 'Q7o': 'fold', 'Q6o': 'fold', 'Q5o': 'fold', 'Q4o': 'fold', 'Q3o': 'fold', 'Q2o': 'fold',
            'J8o': 'fold', 'J7o': 'fold', 'J6o': 'fold', 'J5o': 'fold', 'J4o': 'fold', 'J3o': 'fold', 'J2o': 'fold',
            'T6o': 'fold', 'T5o': 'fold', 'T4o': 'fold', 'T3o': 'fold', 'T2o': 'fold',
            '95o': 0.7, '94o': 0.5, '93o': 'fold', '92o': 'fold',
            '84o': 0.5, '83o': 'fold', '82o': 'fold',
            '73o': 0.5, '72o': 'fold',
            '62o': 'fold',
        }
        
        # Big blind defending range (~60-65% vs button open for 100+ BB)
        self.bb_defending_range_100bb = {
            # Defend with most pairs, suited hands, and high cards
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise', '77': 'raise',
            '66': 'raise', '55': 'raise', '44': 'raise', '33': 'raise', '22': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'A9s': 'call', 'A8s': 'call', 'A7s': 'call', 'A6s': 'call', 'A5s': 'call',
            'A4s': 'call', 'A3s': 'call', 'A2s': 'call',
            'KQs': 'raise', 'KQo': 'call',
            'KJs': 'call', 'KJo': 'call',
            'KTs': 'call', 'K9s': 'call',
            'QJs': 'call', 'QJo': 'call',
            'QTs': 'call', 'Q9s': 'call',
            'JTs': 'call', 'J9s': 'call',
            'T9s': 'call', '98s': 'call', '87s': 'call',
            '76s': 'call', '65s': 'call', '54s': 'call',
        }
        
        # Button 3-bet range (~10-12% vs big blind for 100+ BB)
        # Uses mixed strategies to achieve optimal frequency
        self.button_3bet_range_100bb = {
            # Premium hands - always 3-bet (value)
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            
            # Strong hands - 3-bet for value (always)
            'TT': 'raise', '99': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise',
            'KQs': 'raise',
            
            # Medium hands - 3-bet as bluffs with mixed strategy
            'A5s': 0.85,  # 3-bet 85% of the time as bluff
            'A4s': 0.75,  # 3-bet 75% of the time as bluff
            'KJs': 0.75,  # 3-bet 75% of the time as bluff
            'QJs': 0.65,  # 3-bet 65% of the time as bluff
            'JTs': 0.55,  # 3-bet 55% of the time as bluff
        }
        
        # Big blind 3-bet range (~8-10% vs button open for 100+ BB)
        # Uses mixed strategies to achieve optimal frequency
        self.bb_3bet_range_100bb = {
            # Premium hands - always 3-bet (value)
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            
            # Strong hands - 3-bet for value (always)
            'TT': 'raise', '99': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise',
            'KQs': 'raise',
            
            # Bluffs - use mixed strategy (3-bet with frequency to balance range)
            'A5s': 0.7,  # 3-bet 70% of the time as bluff
            'KJs': 0.6,  # 3-bet 60% of the time as bluff
            'A4s': 0.5,  # 3-bet 50% of the time as bluff
        }
        
        # 4-bet ranges (very narrow, premium hands only for 100+ BB)
        self.button_4bet_range_100bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
        
        self.bb_4bet_range_100bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
        
        # Big blind opening range (rare but possible scenario)
        self.bb_opening_range = {
            # Very tight range - only premium hands
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
        }

        # Small blind opening range for HUNL - Mixed strategy approach
        # Based on specific frequency requirements for optimal GTO play
        self.sb_opening_range_100bb = {
            # 100% raise: All pocket pairs (22-AA)
            '22': 'raise', '33': 'raise', '44': 'raise', '55': 'raise', '66': 'raise',
            '77': 'raise', '88': 'raise', '99': 'raise', 'TT': 'raise', 'JJ': 'raise',
            'QQ': 'raise', 'KK': 'raise', 'AA': 'raise',

            # 100% raise: All suited hands
            '32s': 'raise', '42s': 'raise', '43s': 'raise', '52s': 'raise', '53s': 'raise',
            '54s': 'raise', '62s': 'raise', '63s': 'raise', '64s': 'raise', '65s': 'raise',
            '72s': 'raise', '73s': 'raise', '74s': 'raise', '75s': 'raise', '76s': 'raise',
            '82s': 'raise', '83s': 'raise', '84s': 'raise', '85s': 'raise', '86s': 'raise', '87s': 'raise',
            '92s': 'raise', '93s': 'raise', '94s': 'raise', '95s': 'raise', '96s': 'raise', '97s': 'raise', '98s': 'raise',
            'T2s': 'raise', 'T3s': 'raise', 'T4s': 'raise', 'T5s': 'raise', 'T6s': 'raise', 'T7s': 'raise', 'T8s': 'raise', 'T9s': 'raise',
            'J2s': 'raise', 'J3s': 'raise', 'J4s': 'raise', 'J5s': 'raise', 'J6s': 'raise', 'J7s': 'raise', 'J8s': 'raise', 'J9s': 'raise', 'JTs': 'raise',
            'Q2s': 'raise', 'Q3s': 'raise', 'Q4s': 'raise', 'Q5s': 'raise', 'Q6s': 'raise', 'Q7s': 'raise', 'Q8s': 'raise', 'Q9s': 'raise', 'QJs': 'raise', 'QTs': 'raise',
            'K2s': 'raise', 'K3s': 'raise', 'K4s': 'raise', 'K5s': 'raise', 'K6s': 'raise', 'K7s': 'raise', 'K8s': 'raise', 'K9s': 'raise', 'KJs': 'raise', 'KQs': 'raise', 'KTs': 'raise',
            'A2s': 'raise', 'A3s': 'raise', 'A4s': 'raise', 'A5s': 'raise', 'A6s': 'raise', 'A7s': 'raise', 'A8s': 'raise', 'A9s': 'raise', 'AJs': 'raise', 'AQs': 'raise', 'AKs': 'raise', 'ATs': 'raise',

            # 100% raise: All hands with Ace (offsuit ones not covered by suited rule above)
            'A2o': 'raise', 'A3o': 'raise', 'A4o': 'raise', 'A5o': 'raise', 'A6o': 'raise',
            'A7o': 'raise', 'A8o': 'raise', 'A9o': 'raise', 'AJo': 'raise', 'AQo': 'raise',
            'AKo': 'raise', 'ATo': 'raise',

            # 100% raise: All other broadway offsuit hands not covered above
            'KJo': 'raise', 'KQo': 'raise', 'KTo': 'raise',
            'QJo': 'raise', 'QTo': 'raise',
            'JTo': 'raise',

            # Mixed strategy hands (frequency-based)
            'T3o': 0.85,  # Raise 85% of the time
            '94o': 0.8,   # Raise 80% of the time
            '43o': 0.7,   # Raise 70% of the time
            '53o': 0.7,   # Raise 70% of the time
            '63o': 0.7,   # Raise 70% of the time

            # 100% fold: Specific low offsuit hands
            '32o': 'fold', '42o': 'fold', '52o': 'fold', '62o': 'fold', '72o': 'fold',
            '73o': 'fold', '82o': 'fold', '83o': 'fold', '92o': 'fold', '93o': 'fold', 'T2o': 'fold',
        }
        
        # Button vs big blind 3-bet defending range
        self.button_defending_3bet_range = {
            # Defend against 3-bet with premium hands
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'JJ': 'call', 'TT': 'call',
            'AQs': 'call', 'AQo': 'call',
        }
    
    def _initialize_stack_depth_ranges(self):
        """Initialize ranges for different stack depths (20-30 BB, 50 BB)."""
        # Button opening range for 50 BB (~70-80% of hands)
        self.button_opening_range_50bb = {
            # Premium hands - always open
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            
            # Strong hands - always open
            'TT': 'raise', '99': 'raise', '88': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'KQs': 'raise', 'KQo': 'raise',
            
            # Medium hands - open most of the time
            '77': 'raise', '66': 'raise', '55': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'KJs': 'raise', 'KJo': 'raise',
            'QJs': 'raise', 'QJo': 'raise',
            'JTs': 'raise', 'JTo': 'raise',
            
            # Weak hands - open more liberally than 100 BB
            '44': 'raise', '33': 'raise', '22': 'raise',
            'A9s': 'raise', 'A8s': 'raise', 'A7s': 'raise', 'A6s': 'raise', 'A5s': 'raise',
            'A4s': 'raise', 'A3s': 'raise', 'A2s': 'raise',
            'KTs': 'raise', 'K9s': 'raise',
            'QTs': 'raise', 'Q9s': 'raise',
            'J9s': 'raise', 'T9s': 'raise', '98s': 'raise', '87s': 'raise',
            '76s': 'raise', '65s': 'raise', '54s': 'raise', '43s': 'raise', '42s': 'raise', '32s': 'raise',
            'A9o': 'raise', 'A8o': 'raise',  # More offsuit aces
            'KTo': 'raise', 'K9o': 'raise',
            'QTo': 'raise', 'Q9o': 'raise',
        }
        
        # Button opening range for 20-30 BB (~85-90% of hands - very wide)
        self.button_opening_range_20_30bb = {
            # Open almost everything except trash
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise', '77': 'raise',
            '66': 'raise', '55': 'raise', '44': 'raise', '33': 'raise', '22': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'A9s': 'raise', 'A9o': 'raise',
            'A8s': 'raise', 'A8o': 'raise',
            'A7s': 'raise', 'A7o': 'raise',
            'A6s': 'raise', 'A6o': 'raise',
            'A5s': 'raise', 'A5o': 'raise',
            'A4s': 'raise', 'A4o': 'raise',
            'A3s': 'raise', 'A3o': 'raise',
            'A2s': 'raise', 'A2o': 'raise',
            'KQs': 'raise', 'KQo': 'raise',
            'KJs': 'raise', 'KJo': 'raise',
            'KTs': 'raise', 'KTo': 'raise',
            'K9s': 'raise', 'K9o': 'raise',
            'QJs': 'raise', 'QJo': 'raise',
            'QTs': 'raise', 'QTo': 'raise',
            'Q9s': 'raise', 'Q9o': 'raise',
            'JTs': 'raise', 'JTo': 'raise',
            'J9s': 'raise', 'J9o': 'raise',
            'T9s': 'raise', 'T9o': 'raise',
            '98s': 'raise', '98o': 'raise',
            '87s': 'raise', '87o': 'raise',
            '76s': 'raise', '76o': 'raise',
            '65s': 'raise', '65o': 'raise',
            '54s': 'raise', '54o': 'raise',
            '43s': 'raise', '43o': 'raise',
            '42s': 'raise', '42o': 'raise',
            '32s': 'raise', '32o': 'raise',
        }
        
        # Big blind defending range for 50 BB (~60-70% vs button open)
        self.bb_defending_range_50bb = {
            # Similar to 100 BB but slightly wider
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise', '77': 'raise',
            '66': 'raise', '55': 'raise', '44': 'raise', '33': 'raise', '22': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'A9s': 'call', 'A8s': 'call', 'A7s': 'call', 'A6s': 'call', 'A5s': 'call',
            'A4s': 'call', 'A3s': 'call', 'A2s': 'call',
            'KQs': 'raise', 'KQo': 'call',
            'KJs': 'call', 'KJo': 'call',
            'KTs': 'call', 'K9s': 'call',
            'QJs': 'call', 'QJo': 'call',
            'QTs': 'call', 'Q9s': 'call',
            'JTs': 'call', 'J9s': 'call',
            'T9s': 'call', '98s': 'call', '87s': 'call',
            '76s': 'call', '65s': 'call', '54s': 'call',
        }
        
        # Big blind defending range for 20-30 BB (~70-75% vs button open - wider)
        self.bb_defending_range_20_30bb = {
            # Much wider defending range
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise', '77': 'raise',
            '66': 'raise', '55': 'raise', '44': 'raise', '33': 'raise', '22': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'ATs': 'raise', 'ATo': 'raise',
            'A9s': 'call', 'A9o': 'call',
            'A8s': 'call', 'A8o': 'call',
            'A7s': 'call', 'A7o': 'call',
            'A6s': 'call', 'A6o': 'call',
            'A5s': 'call', 'A5o': 'call',
            'A4s': 'call', 'A4o': 'call',
            'A3s': 'call', 'A3o': 'call',
            'A2s': 'call', 'A2o': 'call',
            'KQs': 'raise', 'KQo': 'call',
            'KJs': 'call', 'KJo': 'call',
            'KTs': 'call', 'KTo': 'call',
            'K9s': 'call', 'K9o': 'call',
            'QJs': 'call', 'QJo': 'call',
            'QTs': 'call', 'QTo': 'call',
            'Q9s': 'call', 'Q9o': 'call',
            'JTs': 'call', 'JTo': 'call',
            'J9s': 'call', 'J9o': 'call',
            'T9s': 'call', 'T9o': 'call',
            '98s': 'call', '98o': 'call',
            '87s': 'call', '87o': 'call',
            '76s': 'call', '76o': 'call',
            '65s': 'call', '65o': 'call',
            '54s': 'call', '54o': 'call',
        }
        
        # Button 3-bet range for 50 BB (~12-15% vs big blind)
        # Uses mixed strategies for optimal frequency
        self.button_3bet_range_50bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise', '99': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise',
            'KQs': 'raise',
            'A5s': 0.8,  # 3-bet 80% of the time
            'A4s': 0.7,  # 3-bet 70% of the time
            'KJs': 0.7,  # 3-bet 70% of the time
            'QJs': 0.6,  # 3-bet 60% of the time
            'JTs': 0.5,  # 3-bet 50% of the time
        }
        
        # Button 3-bet range for 20-30 BB (~15-18% vs big blind - wider due to stack depth)
        # Uses mixed strategies for optimal frequency
        self.button_3bet_range_20_30bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'KQs': 'raise', 'KQo': 'raise',
            'A5s': 0.85,  # 3-bet 85% of the time
            'A4s': 0.75,  # 3-bet 75% of the time
            'A3s': 0.65,  # 3-bet 65% of the time
            'KJs': 0.75,  # 3-bet 75% of the time
            'KJo': 0.65,  # 3-bet 65% of the time
            'QJs': 0.7,   # 3-bet 70% of the time
            'QJo': 0.6,   # 3-bet 60% of the time
            'JTs': 0.65,  # 3-bet 65% of the time
            'JTo': 0.55,  # 3-bet 55% of the time
            'T9s': 0.6,   # 3-bet 60% of the time
        }
        
        # Big blind 3-bet range for 50 BB (~10-12% vs button open)
        # Uses mixed strategies for optimal frequency
        self.bb_3bet_range_50bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise', '99': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise',
            'KQs': 'raise',
            'A5s': 0.75,  # 3-bet 75% of the time
            'KJs': 0.65,  # 3-bet 65% of the time
            'A4s': 0.55,  # 3-bet 55% of the time
        }
        
        # Big blind 3-bet range for 20-30 BB (~12-15% vs button open - wider due to stack depth)
        # Uses mixed strategies for optimal frequency
        self.bb_3bet_range_20_30bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise', '99': 'raise', '88': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise', 'AJo': 'raise',
            'KQs': 'raise', 'KQo': 'raise',
            'A5s': 0.8,  # 3-bet 80% of the time
            'A4s': 0.7,  # 3-bet 70% of the time
            'KJs': 0.7,  # 3-bet 70% of the time
            'KJo': 0.6,  # 3-bet 60% of the time
            'QJs': 0.6,  # 3-bet 60% of the time
        }
        
        # 4-bet ranges for 50 BB (~5-8% for button, ~4-6% for BB)
        self.button_4bet_range_50bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'JJ': 'raise',
        }
        
        self.bb_4bet_range_50bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
        
        # 4-bet ranges for 20-30 BB (all-in or fold - very narrow)
        self.button_4bet_range_20_30bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
        
        self.bb_4bet_range_20_30bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
    
    def _initialize_allin_ranges(self):
        """Initialize all-in ranges for different stack depths."""
        # All-in range for 20-30 BB (wider - premium + some bluffs)
        self.allin_range_20_30bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise', 'JJ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
            'TT': 'raise',
            'AQs': 'raise', 'AQo': 'raise',
            'AJs': 'raise',
            'KQs': 'raise',
        }
        
        # All-in range for 50 BB (very premium hands only)
        self.allin_range_50bb = {
            'AA': 'raise', 'KK': 'raise', 'QQ': 'raise',
            'AKs': 'raise', 'AKo': 'raise',
        }
        
        # All-in range for 100+ BB (extremely premium - AA, KK only)
        self.allin_range_100bb = {
            'AA': 'raise', 'KK': 'raise',
        }
    
    def _get_stack_depth_category(self, stack_depth):
        """
        Categorize stack depth into ranges.
        
        Args:
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            str: '20_30', '50', or '100'
        """
        if stack_depth < 20:
            return '20_30'  # Very short stack
        elif stack_depth < 30:
            return '20_30'  # Short stack
        elif stack_depth < 50:
            return '50'  # Medium stack
        elif stack_depth < 100:
            return '50'  # Medium-deep stack
        else:
            return '100'  # Deep stack (100+ BB)
    
    def _get_range_for_stack_depth(self, position, action_type, stack_depth):
        """
        Get the appropriate range dictionary for given position, action type, and stack depth.
        
        Args:
            position (str): "button" or "big_blind"
            action_type (str): "open", "defend", "3bet", "4bet"
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: Range dictionary
        """
        import logging
        logger = logging.getLogger(__name__)
        
        category = self._get_stack_depth_category(stack_depth)
        logger.debug(f"[GTO RULES] _get_range_for_stack_depth called - position={position}, action_type={action_type}, stack_depth={stack_depth:.1f}, category={category}")
        
        if position == 'button':
            if action_type == 'open':
                # Use SB opening range for button opening (HUNL SB opening)
                if category == '100':
                    return self.sb_opening_range_100bb
                elif category == '20_30':
                    return self.button_opening_range_20_30bb
                elif category == '50':
                    return self.button_opening_range_50bb
                else:  # 100+
                    return self.button_opening_range_100bb
            elif action_type == 'defend':
                # Button defending vs BB 3-bet
                return self.button_defending_3bet_range
            elif action_type == '3bet':
                if category == '20_30':
                    return self.button_3bet_range_20_30bb
                elif category == '50':
                    return self.button_3bet_range_50bb
                else:  # 100+
                    return self.button_3bet_range_100bb
            elif action_type == '4bet':
                if category == '20_30':
                    return self.button_4bet_range_20_30bb
                elif category == '50':
                    return self.button_4bet_range_50bb
                else:  # 100+
                    return self.button_4bet_range_100bb
        elif position == 'big_blind':
            if action_type == 'open':
                logger.debug(f"[GTO RULES] Returning bb_opening_range (very tight, all 'raise')")
                return self.bb_opening_range
            elif action_type == 'defend':
                if category == '20_30':
                    logger.debug(f"[GTO RULES] Returning bb_defending_range_20_30bb (has 'call' entries)")
                    return self.bb_defending_range_20_30bb
                elif category == '50':
                    logger.debug(f"[GTO RULES] Returning bb_defending_range_50bb (has 'call' entries)")
                    return self.bb_defending_range_50bb
                else:  # 100+
                    logger.debug(f"[GTO RULES] Returning bb_defending_range_100bb (has 'call' entries - this is the correct range for facing button open!)")
                    return self.bb_defending_range_100bb
            elif action_type == '3bet':
                if category == '20_30':
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_3bet_range_20_30bb (mostly 'raise', few/no 'call' entries)")
                    return self.bb_3bet_range_20_30bb
                elif category == '50':
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_3bet_range_50bb (mostly 'raise', few/no 'call' entries)")
                    return self.bb_3bet_range_50bb
                else:  # 100+
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_3bet_range_100bb (mostly 'raise', few/no 'call' entries - if action_type should be 'defend', this is wrong!)")
                    return self.bb_3bet_range_100bb
            elif action_type == '4bet':
                if category == '20_30':
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_4bet_range_20_30bb (all 'raise', no 'call' entries)")
                    return self.bb_4bet_range_20_30bb
                elif category == '50':
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_4bet_range_50bb (all 'raise', no 'call' entries)")
                    return self.bb_4bet_range_50bb
                else:  # 100+
                    logger.warning(f"[GTO RULES] ⚠️ Returning bb_4bet_range_100bb (all 'raise', no 'call' entries - if action_type should be 'defend', this is wrong!)")
                    return self.bb_4bet_range_100bb
        
        logger.error(f"[GTO RULES] No range found for position={position}, action_type={action_type}, category={category}, returning empty dict")
        return {}
    
    def _get_allin_range(self, stack_depth):
        """
        Get all-in range for given stack depth.
        
        Args:
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: All-in range dictionary
        """
        category = self._get_stack_depth_category(stack_depth)
        
        if category == '20_30':
            return self.allin_range_20_30bb
        elif category == '50':
            return self.allin_range_50bb
        else:  # 100+
            return self.allin_range_100bb
    
    def get_preflop_action(self, hand, position, action_type, stack_depth=100, use_frequency=True):
        """
        Get optimal preflop action for a hand.
        
        Args:
            hand (str): Hand in string format (e.g., "AA", "AKs", "72o")
            position (str): "button" or "big_blind"
            action_type (str): "open", "defend", "3bet", "4bet", "allin"
            stack_depth (int): Stack depth in big blinds (default 100)
            use_frequency (bool): If True, use random selection for frequency-based actions (default True)
        
        Returns:
            str: Optimal action ("raise", "call", "fold")
            If use_frequency=False and hand has frequency, returns the frequency value (float)
        """
        try:
            # Validate inputs
            if not hand or not isinstance(hand, str):
                return 'fold'
            
            if position not in ['button', 'big_blind']:
                return 'fold'
            
            if action_type not in ['open', 'defend', '3bet', '4bet', 'allin']:
                return 'fold'
            
            # Handle invalid stack depth
            if stack_depth < 0 or stack_depth > 1000:
                # Use closest available range
                if stack_depth < 0:
                    stack_depth = 20
                else:
                    stack_depth = 100
            
            # Handle all-in scenarios
            if action_type == 'allin':
                allin_range = self._get_allin_range(stack_depth)
                action_value = allin_range.get(hand, 'fold')
                # All-in ranges don't use frequencies, so return directly
                return action_value
            
            # Get appropriate range for stack depth
            range_dict = self._get_range_for_stack_depth(position, action_type, stack_depth)
            
            # DEBUG: Log range lookup with statistics
            import logging
            logger = logging.getLogger(__name__)
            range_size = len(range_dict) if range_dict else 0
            
            # Count action types in range for debugging
            raise_count = sum(1 for v in range_dict.values() if v == 'raise')
            call_count = sum(1 for v in range_dict.values() if v == 'call')
            fold_count = sum(1 for v in range_dict.values() if v == 'fold')
            freq_count = sum(1 for v in range_dict.values() if isinstance(v, (int, float)) and 0.0 < v <= 1.0)
            
            logger.debug(f"[GTO RULES] ===== RANGE LOOKUP =====")
            logger.debug(f"[GTO RULES] Position: {position}, Action type: {action_type}, Stack depth: {stack_depth}")
            logger.debug(f"[GTO RULES] Range size: {range_size} hands")
            logger.debug(f"[GTO RULES] Range composition - Raise: {raise_count}, Call: {call_count}, Fold: {fold_count}, Frequency: {freq_count}")
            logger.debug(f"[GTO RULES] Looking up hand: {hand}")
            
            if not range_dict:
                logger.warning(f"[GTO RULES] No range found for {position}/{action_type}/{stack_depth}, returning 'fold'")
                logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: FOLD (no range) =====")
                return 'fold'
            
            action_value = range_dict.get(hand, 'fold')
            logger.debug(f"[GTO RULES] Hand {hand} lookup result: {action_value} (type: {type(action_value).__name__})")

            if hand not in range_dict:
                logger.debug(f"[GTO RULES] Hand {hand} NOT in range, defaulting to 'fold'")
                logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: FOLD (not in range) =====")
                return 'fold'
            
            # Handle frequency-based actions (mixed strategies)
            if isinstance(action_value, (int, float)) and 0.0 < action_value <= 1.0:
                # This is a frequency (e.g., 0.7 means raise 70% of the time)
                logger.debug(f"[GTO RULES] Frequency-based action: {action_value} ({action_value*100:.1f}% to raise)")
                if use_frequency:
                    # Use random selection based on frequency
                    # Import random at module level to ensure proper state management
                    import random
                    # Generate random number for frequency-based decision
                    rand_val = random.random()
                    logger.debug(f"[GTO RULES] Random value: {rand_val:.3f}, threshold: {action_value:.3f}")
                    if rand_val < action_value:
                        logger.debug(f"[GTO RULES] Frequency check passed: returning 'raise'")
                        logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: RAISE (frequency) =====")
                        return 'raise'
                    else:
                        # If not raising, default to fold for 3bet/4bet scenarios
                        # or call for defend scenarios
                        if action_type == 'defend':
                            logger.debug(f"[GTO RULES] Frequency check failed (defend): returning 'call'")
                            logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: CALL (frequency) =====")
                            return 'call'
                        else:
                            logger.debug(f"[GTO RULES] Frequency check failed ({action_type}): returning 'fold'")
                            logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: FOLD (frequency) =====")
                            return 'fold'
                else:
                    # Return the frequency value for caller to handle
                    logger.debug(f"[GTO RULES] use_frequency=False, returning frequency value: {action_value}")
                    logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: FREQUENCY VALUE =====")
                    return action_value
            
            # Handle string actions ('raise', 'call', 'fold')
            logger.debug(f"[GTO RULES] String action: returning '{action_value}'")
            logger.debug(f"[GTO RULES] ===== END RANGE LOOKUP: {action_value.upper()} =====")
            return action_value
        except Exception:
            # Error handling - default to fold
            return 'fold'
    
    def calculate_pot_odds(self, pot, bet):
        """
        Calculate pot odds as a decimal ratio.
        
        Formula: (pot + bet) / bet
        
        Args:
            pot (float): Current pot size
            bet (float): Amount to call
        
        Returns:
            float: Pot odds as decimal ratio (e.g., 3.0 means 3:1), or None if bet is 0
        """
        if bet == 0:
            return None  # Check scenario, no pot odds calculation needed
        
        if pot < 0 or bet < 0:
            return None  # Invalid values
        
        return (pot + bet) / bet
    
    def pot_odds_to_percentage(self, pot_odds):
        """
        Convert pot odds ratio to percentage.
        
        Args:
            pot_odds (float): Pot odds as decimal ratio
        
        Returns:
            float: Percentage (0-100)
        """
        if pot_odds is None:
            return None
        
        return (1 / (pot_odds + 1)) * 100
    
    def adjust_range_for_stack_depth(self, base_range, stack_depth):
        """
        Adjust a base range based on stack depth using multipliers.
        
        This method can be used to dynamically adjust ranges for intermediate
        stack depths or to apply fine-grained adjustments.
        
        Args:
            base_range (dict): Base range dictionary (hand -> action)
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: Adjusted range dictionary
        """
        if not base_range:
            return {}
        
        category = self._get_stack_depth_category(stack_depth)
        
        # Define multipliers for range adjustments
        # Short stacks: widen ranges (more hands included)
        # Deep stacks: tighten ranges (fewer hands included)
        multipliers = {
            '20_30': 1.15,  # Widen by 15% (more hands can raise/call)
            '50': 1.0,      # No adjustment (base range)
            '100': 0.95     # Tighten by 5% (fewer hands)
        }
        
        multiplier = multipliers.get(category, 1.0)
        
        # For intermediate stack depths, interpolate
        if category == '20_30' and 20 <= stack_depth < 30:
            # Interpolate between 20 BB (1.15) and 30 BB (1.0)
            progress = (stack_depth - 20) / 10
            multiplier = 1.15 - (0.15 * progress)
        elif category == '50' and 30 <= stack_depth < 50:
            # Interpolate between 30 BB (1.0) and 50 BB (1.0)
            multiplier = 1.0
        elif category == '50' and 50 <= stack_depth < 100:
            # Interpolate between 50 BB (1.0) and 100 BB (0.95)
            progress = (stack_depth - 50) / 50
            multiplier = 1.0 - (0.05 * progress)
        
        # Apply multiplier to range
        # Since ranges are discrete (hand -> action), we can't directly multiply
        # Instead, we adjust the threshold for marginal hands
        adjusted_range = base_range.copy()
        
        # For now, return the base range as-is since we use pre-defined ranges
        # This method provides a framework for future dynamic adjustments
        return adjusted_range
    
    def get_bet_sizing_guidelines(self, action_type, stack_depth=100):
        """
        Get bet sizing guidelines for preflop actions.
        
        Args:
            action_type (str): "open", "3bet", "4bet"
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: {
                'min_size': float,  # Minimum bet size (in big blinds)
                'max_size': float,  # Maximum bet size (in big blinds)
                'optimal_size': float  # Optimal bet size (in big blinds)
            }
        """
        category = self._get_stack_depth_category(stack_depth)
        
        # Base sizing guidelines (in big blinds)
        if action_type == 'open':
            if category == '20_30':
                # Shorter stacks: smaller opening sizes acceptable
                return {'min_size': 2.0, 'max_size': 2.5, 'optimal_size': 2.2}
            elif category == '50':
                return {'min_size': 2.0, 'max_size': 3.0, 'optimal_size': 2.5}
            else:  # 100+
                return {'min_size': 2.0, 'max_size': 3.0, 'optimal_size': 2.5}
        
        elif action_type == '3bet':
            # 3-bet sizing: 2.5-3x of the open
            if category == '20_30':
                return {'min_size': 5.0, 'max_size': 7.5, 'optimal_size': 6.0}
            elif category == '50':
                return {'min_size': 6.0, 'max_size': 9.0, 'optimal_size': 7.5}
            else:  # 100+
                return {'min_size': 6.0, 'max_size': 9.0, 'optimal_size': 7.5}
        
        elif action_type == '4bet':
            # 4-bet sizing: 2-2.5x of the 3-bet
            if category == '20_30':
                # Short stacks: often all-in
                return {'min_size': 12.0, 'max_size': 20.0, 'optimal_size': 15.0}
            elif category == '50':
                return {'min_size': 12.0, 'max_size': 20.0, 'optimal_size': 15.0}
            else:  # 100+
                return {'min_size': 12.0, 'max_size': 22.5, 'optimal_size': 15.0}
        
        return {'min_size': 0, 'max_size': 0, 'optimal_size': 0}

