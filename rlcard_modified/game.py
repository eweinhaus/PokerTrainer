"""
Modified RLCard NolimitholdemGame with BB-first action order support
Based on RLCard v1.2.0
"""

import numpy as np
from copy import deepcopy
from enum import Enum
from rlcard.games.base import Card
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem.player import NolimitholdemPlayer as Player
from rlcard.games.nolimitholdem.dealer import Dealer
from rlcard.games.nolimitholdem.judger import Judger
from rlcard.games.nolimitholdem import Round, Action


class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class NolimitholdemGame:
    """Modified No Limit Hold'em Game with BB-first action order support"""

    def __init__(self, allow_step_back=False, num_players=2, enable_bb_first_action_order=False):
        """Initialize the class no limit holdem Game

        Args:
            allow_step_back (bool): Whether to allow stepping back
            num_players (int): Number of players
            enable_bb_first_action_order (bool): Enable BB-first action order for heads-up games
        """
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.enable_bb_first_action_order = enable_bb_first_action_order

        self.np_random = np.random.RandomState()

        # small blind and big blind
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # config players
        self.init_chips = [100] * num_players

        # If None, the dealer will be randomly chosen
        self.dealer_id = None

    def init_game(self):
        """
        Initialize the game of not limit holdem

        This version supports two-player no limit texas holdem

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        """
        if self.dealer_id is None:
            self.dealer_id = self.np_random.randint(0, self.num_players)

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize players to play the game
        self.players = [Player(i, self.init_chips[i], self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        # Deal cards to each  player to prepare for the first round
        for i in range(2 * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        # Initialize public cards
        self.public_cards = []
        self.stage = Stage.PREFLOP

        # Big blind and small blind
        s = (self.dealer_id + 1) % self.num_players
        b = (self.dealer_id + 2) % self.num_players
        self.players[b].bet(chips=self.big_blind)
        self.players[s].bet(chips=self.small_blind)

        # The player next to the big blind plays the first
        self.game_pointer = (b + 1) % self.num_players

        # Initialize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(self.num_players, self.big_blind, dealer=self.dealer, np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # Save the history for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def step(self, action):
        """
        Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next player id
        """

        if action not in self.get_legal_actions():
            print(action, self.get_legal_actions())
            print(self.get_state(self.game_pointer))
            raise Exception('Action not allowed')

        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            self.history.append((r, b, r_c, d, p, ps))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        players_in_bypass = [1 if player.status in (PlayerStatus.FOLDED, PlayerStatus.ALLIN) else 0 for player in self.players]
        if self.num_players - sum(players_in_bypass) == 1:
            last_player = players_in_bypass.index(0)
            if self.round.raised[last_player] >= max(self.round.raised):
                # If the last player has put enough chips, he is also bypassed
                players_in_bypass[last_player] = 1

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # MODIFIED: BB-first action order for heads-up games
            if self.enable_bb_first_action_order and self.num_players == 2 and self.round_counter >= 1:
                # Postflop streets: BB (dealer) acts first
                self.game_pointer = self.dealer_id
            else:
                # Original logic: SB (dealer + 1) acts first for preflop and all multiplayer games
                self.game_pointer = (self.dealer_id + 1) % self.num_players

            if sum(players_in_bypass) < self.num_players:
                while players_in_bypass[self.game_pointer]:
                    self.game_pointer = (self.game_pointer + 1) % self.num_players

            # For the first round, we deal 3 cards
            if self.round_counter == 0:
                self.stage = Stage.FLOP
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            # For the following rounds, we deal only 1 card
            if self.round_counter == 1:
                self.stage = Stage.TURN
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            if self.round_counter == 2:
                self.stage = Stage.RIVER
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    # Include other methods from original NolimitholdemGame
    def get_state(self, player_id):
        """Get the state of the game for a specific player"""
        # Basic implementation - would need full implementation from original RLCard
        hand = []
        if player_id < len(self.players):
            hand = [card.get_index() for card in self.players[player_id].hand]

        public_cards = [card.get_index() for card in self.public_cards]

        # Calculate pot and raised amounts
        pot = 0
        raised = []
        for player in self.players:
            pot += player.in_chips
            raised.append(player.in_chips)

        return {
            'raw_obs': {
                'stage': self.stage.value if hasattr(self.stage, 'value') else self.stage,
                'public_cards': public_cards,
                'hand': hand,
                'pot': pot,
                'raised': raised,
                'current_player': self.game_pointer
            },
            'raw_legal_actions': self.get_legal_actions(),
            'action_record': []
        }

    def get_legal_actions(self):
        """Get legal actions for current player"""
        # Basic implementation - would need full round logic from original RLCard
        from rlcard.core import Action
        return [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT, Action.ALL_IN]

    def is_over(self):
        """Check if game is over"""
        return self.round_counter >= 4 or len([p for p in self.players if p.status == PlayerStatus.ACTIVE]) <= 1
