"""
Mock implementation of RLCard for development when the real package isn't available.
This provides minimal functionality to allow the app to start.
"""

import random

class Action:
    """Mock Action class matching RLCard"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4

class MockPlayer:
    def __init__(self, hand):
        self.hand = hand
        self.in_chips = 0

class MockDealer:
    def __init__(self):
        self.pot = 3

class MockGame:
    def __init__(self):
        # Deal random cards instead of hardcoded AA
        hand1, hand2 = self._deal_random_hands()
        self.players = [
            MockPlayer(hand1),
            MockPlayer(hand2)
        ]
        self.dealer = MockDealer()
        self.dealer_id = 0

    def _deal_random_hands(self):
        """Deal two random hole cards for each player"""
        # Create a deck of 52 cards (0-51)
        deck = list(range(52))
        random.shuffle(deck)

        # Deal 2 cards to each player
        return [deck[0], deck[1]], [deck[2], deck[3]]

class MockEnvironment:
    def __init__(self):
        self.num_actions = 5  # Standard poker actions
        self.agents = []
        self.game = MockGame()
        # Initialize with proper game state (random cards, 100 BB stacks)
        self.game_state = self._create_initial_game_state()

    def _create_initial_game_state(self):
        """Create initial game state with random cards and proper stacks"""
        # Deal random hands
        hand1, hand2 = self.game._deal_random_hands()

        return {
            'stage': 0,  # Preflop
            'pot': 3,  # SB + BB in chip units (1 + 2)
            'big_blind': 2,
            'public_cards': [],
            'hands': [hand1, hand2],  # Random hands instead of hardcoded AA
            'raised': [1, 2],  # SB: 1, BB: 2
            'all_chips': [198, 196],  # 100 BB stacks minus blinds (200-2=198, 200-4=196)
            'current_player': 0
        }

    def set_agents(self, agents):
        self.agents = agents

    def reset(self):
        """Reset the game and return initial state"""
        # Redeal cards for new hand
        self.game = MockGame()  # Create new game with new random cards
        self.game_state = self._create_initial_game_state()

        # Return state from player 0's perspective
        state = {
            'raw_obs': {
                'hand': self.game_state['hands'][0],  # Player 0's hand
                'public_cards': self.game_state['public_cards'],
                'pot': self.game_state['pot'],
                'big_blind': 2,
                'all_chips': self.game_state['all_chips'],
                'raised': self.game_state['raised'],
                'stage': self.game_state['stage'],
                'legal_actions': [0, 1, 2, 3, 4],
                'raw_legal_actions': [0, 1, 2, 3, 4]
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }
        return state, 0  # state, current_player

    def step(self, action, raw_action=False):
        """Take an action and return new state"""
        # Simple mock logic - just switch players and update pot
        current_player = self.game_state['current_player']
        opponent = 1 - current_player

        # Store previous bet amount for this player
        previous_bet = self.game_state['raised'][current_player]

        # Update raised amounts based on action
        if action == Action.CHECK_CALL:
            # Match opponent's bet
            self.game_state['raised'][current_player] = self.game_state['raised'][opponent]
        elif action == Action.RAISE_HALF_POT:
            # Preflop: Raise to 3 BB total (standard button open)
            # Postflop: Raise by half pot
            if self.game_state.get('stage', 0) == 0:  # Preflop
                big_blind = self.game_state.get('big_blind', 2)
                # Raise to 3 BB total
                self.game_state['raised'][current_player] = 3 * big_blind
            else:  # Postflop
                # Raise by half pot
                raise_amount = self.game_state['pot'] // 2
                self.game_state['raised'][current_player] = self.game_state['raised'][opponent] + raise_amount
        elif action == Action.RAISE_POT:
            # Preflop: 3-bet to 10 BB total (facing button open)
            # Postflop: Raise by full pot
            if self.game_state.get('stage', 0) == 0:  # Preflop
                big_blind = self.game_state.get('big_blind', 2)
                # 3-bet to 10 BB total
                self.game_state['raised'][current_player] = 10 * big_blind
            else:  # Postflop
                # Raise by full pot
                raise_amount = self.game_state['pot']
                self.game_state['raised'][current_player] = self.game_state['raised'][opponent] + raise_amount
        elif action == Action.ALL_IN:
            # All in
            self.game_state['raised'][current_player] = self.game_state['all_chips'][current_player]

        # Calculate how much this player added to their bet
        bet_amount = self.game_state['raised'][current_player] - previous_bet

        # Update player in_chips
        for i, player in enumerate(self.game.players):
            player.in_chips = self.game_state['raised'][i]

        # Update pot cumulatively in chip units (not BB units)
        # Pot should accumulate across all betting rounds
        if 'cumulative_pot' not in self.game_state:
            self.game_state['cumulative_pot'] = self.game_state.get('pot', 0)

        # Add the additional bet amount to cumulative pot
        self.game_state['cumulative_pot'] += bet_amount
        self.game_state['pot'] = self.game_state['cumulative_pot']

        # Switch to next player
        self.game_state['current_player'] = opponent

        # Return new state from new current player's perspective
        state = {
            'raw_obs': {
                'hand': self.game_state['hands'][opponent],  # New current player's hand
                'public_cards': self.game_state['public_cards'],
                'pot': self.game_state['pot'],
                'big_blind': 2,
                'all_chips': self.game_state['all_chips'],
                'raised': self.game_state['raised'],
                'stage': self.game_state['stage'],
                'legal_actions': [0, 1, 2, 3, 4],
                'raw_legal_actions': [0, 1, 2, 3, 4]
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }
        return state, opponent

    def get_legal_actions(self):
        """Return legal actions for current state"""
        return [0, 1, 2, 3, 4]  # FOLD, CHECK_CALL, RAISE_HALF_POT, RAISE_POT, ALL_IN

    def is_over(self):
        """Check if game is over (mock - always return False for now)"""
        return False

    def get_state(self, player_id):
        """Get state from specific player's perspective"""
        return {
            'raw_obs': {
                'hand': self.game_state['hands'][player_id],
                'public_cards': self.game_state['public_cards'],
                'pot': self.game_state['pot'],
                'big_blind': 2,
                'all_chips': self.game_state['all_chips'],
                'raised': self.game_state['raised'],
                'stage': self.game_state['stage'],
                'legal_actions': [0, 1, 2, 3, 4],
                'raw_legal_actions': [0, 1, 2, 3, 4]
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }

class MockAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = True

    def step(self, state):
        """Mock agent always calls/checks"""
        return Action.CHECK_CALL

def make(game_name):
    """Mock rlcard.make function"""
    if game_name == 'no-limit-holdem':
        return MockEnvironment()
    else:
        raise ValueError(f"Unsupported game: {game_name}")

# Create agents submodule
import types
agents = types.ModuleType('agents')
agents.RandomAgent = MockAgent

# Make agents available as rlcard.agents
import sys
sys.modules['rlcard_mock.agents'] = agents

# Make Action available at package level
sys.modules['rlcard_mock'].Action = Action
