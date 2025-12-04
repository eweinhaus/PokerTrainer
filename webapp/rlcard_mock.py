"""
Mock implementation of RLCard for development when the real package isn't available.
This provides minimal functionality to allow the app to start.
"""

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
        self.players = [
            MockPlayer([0, 13]),  # Player 0: AA
            MockPlayer([26, 39])  # Player 1: KQs
        ]
        self.dealer = MockDealer()
        self.dealer_id = 0

class MockEnvironment:
    def __init__(self):
        self.num_actions = 5  # Standard poker actions
        self.agents = []
        self.game = MockGame()
        self.game_state = {
            'stage': 0,  # Preflop
            'pot': 3,    # SB + BB
            'public_cards': [],
            'hands': [[0, 13], [26, 39]],  # Player 0: AA, Player 1: KQs
            'raised': [1, 2],  # SB: 1, BB: 2
            'all_chips': [99, 98],  # Starting chips minus blinds
            'current_player': 0
        }

    def set_agents(self, agents):
        self.agents = agents

    def reset(self):
        """Reset the game and return initial state"""
        self.game_state = {
            'stage': 0,  # Preflop
            'pot': 3,    # SB + BB
            'public_cards': [],
            'hands': [[0, 13], [26, 39]],  # Player 0: AA, Player 1: KQs
            'raised': [1, 2],  # SB: 1, BB: 2
            'all_chips': [99, 98],  # Starting chips minus blinds
            'current_player': 0
        }

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
                'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
                'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }
        return state, 0  # state, current_player

    def step(self, action):
        """Take an action and return new state"""
        # Simple mock logic - just switch players and update pot
        current_player = self.game_state['current_player']
        opponent = 1 - current_player

        # Update raised amounts based on action
        if action == Action.CHECK_CALL:
            # Match opponent's bet
            self.game_state['raised'][current_player] = self.game_state['raised'][opponent]
        elif action == Action.RAISE_HALF_POT:
            # Raise by half pot
            raise_amount = self.game_state['pot'] // 2
            self.game_state['raised'][current_player] = self.game_state['raised'][opponent] + raise_amount
        elif action == Action.RAISE_POT:
            # Raise by full pot
            raise_amount = self.game_state['pot']
            self.game_state['raised'][current_player] = self.game_state['raised'][opponent] + raise_amount
        elif action == Action.ALL_IN:
            # All in
            self.game_state['raised'][current_player] = self.game_state['all_chips'][current_player]

        # Update player in_chips
        for i, player in enumerate(self.game.players):
            player.in_chips = self.game_state['raised'][i]

        # Update pot
        self.game_state['pot'] = sum(self.game_state['raised'])

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
                'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
                'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }
        return state, opponent

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
                'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
                'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
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
