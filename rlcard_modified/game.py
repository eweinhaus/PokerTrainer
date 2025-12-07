"""
Modified RLCard methods for BB-first action order support
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
    """Modified methods for No Limit Hold'em Game with BB-first action order support"""

    def init_game(self):
        """Initialize the game with BB-first action order support for heads-up games."""
        # Initialize players
        self.players = [Player(i, self.init_chips[i]) for i in range(self.num_players)]

        # Initialize dealer and shuffle
        self.dealer = Dealer(self.np_random)

        # Initialize judger
        self.judger = Judger(self.np_random)

        # Initialize public cards
        self.public_cards = []

        # Initialize round
        self.round = Round(raise_amount=self.big_blind, allowed_raise_num=0)

        # Initialize game state
        self.game_pointer = 0
        self.round_counter = 0
        self.history = []
        self.stage = Stage.PREFLOP

        # Initialize dealer id
        if self.dealer_id is None:
            self.dealer_id = self.np_random.choice(self.num_players)
        else:
            self.dealer_id = self.dealer_id % self.num_players

        # Deal hole cards
        for i in range(self.num_players):
            self.players[i].hand = self.dealer.deal_card(2)

        # Post blinds
        small_blind_pos = (self.dealer_id + 1) % self.num_players
        big_blind_pos = (self.dealer_id + 2) % self.num_players

        self.players[small_blind_pos].bet(chips=self.small_blind)
        self.players[big_blind_pos].bet(chips=self.big_blind)

        # MODIFIED: BB-first action order for heads-up games
        if self.num_players == 2:
            # Heads-up: SB acts first preflop, BB acts first postflop
            self.game_pointer = small_blind_pos  # SB acts first preflop
            self.enable_bb_first_action_order = True
            print("BB-first action order enabled for heads-up game")
        else:
            # Multiplayer: SB acts first
            self.game_pointer = small_blind_pos

        self.round.start_new_round(self.game_pointer)
        state = self.get_state(self.game_pointer)

        return state, self.game_pointer
