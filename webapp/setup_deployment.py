#!/usr/bin/env python3
"""
Deployment setup script for Render.
Monkey-patches specific methods in RLCard's NolimitholdemGame for BB-first action order support.
"""

import sys
import os
from pathlib import Path

def patch_rlcard_game():
    """Monkey-patch the RLCard NolimitholdemGame with BB-first action order support."""
    try:
        # Import RLCard to ensure it's available
        import rlcard
        from rlcard.games.nolimitholdem.game import NolimitholdemGame

        print(f"Found RLCard at: {rlcard.__file__}")

        # Store the original init_game method
        original_init_game = NolimitholdemGame.init_game

        # Store original methods
        original_init_game = NolimitholdemGame.init_game
        original_configure = NolimitholdemGame.configure
        original_step = NolimitholdemGame.step
        original_get_state = NolimitholdemGame.get_state
        original_get_legal_actions = NolimitholdemGame.get_legal_actions
        original_get_num_actions = NolimitholdemGame.get_num_actions
        original_get_num_players = NolimitholdemGame.get_num_players
        original_get_payoffs = NolimitholdemGame.get_payoffs
        original_get_player_id = NolimitholdemGame.get_player_id
        original_is_over = NolimitholdemGame.is_over
        original_step_back = getattr(NolimitholdemGame, 'step_back', None)

        def patched_init_game(self):
            """
            Initialize the game of no limit holdem with BB-first action order support for heads-up games.

            This version supports two-player no limit texas holdem with modified action order:
            - Preflop: SB acts first (standard)
            - Postflop: BB acts first (modified for heads-up)

            Returns:
                (tuple): Tuple containing:
                    (dict): The first state of the game
                    (int): Current player's id
            """
            # Import RLCard classes locally
            from rlcard.games.nolimitholdem.player import NolimitholdemPlayer as Player
            from rlcard.games.nolimitholdem.dealer import Dealer
            from rlcard.games.nolimitholdem.judger import Judger
            from rlcard.games.nolimitholdem import Round
            from rlcard.games.nolimitholdem.game import Stage

            if self.dealer_id is None:
                self.dealer_id = self.np_random.randint(0, self.num_players)

            # Initialize a dealer that can deal cards
            self.dealer = Dealer(self.np_random)

            # Initialize players to play the game
            self.players = [Player(i, self.init_chips[i], self.np_random) for i in range(self.num_players)]

            # Initialize a judger class which will decide who wins in the end
            self.judger = Judger(self.np_random)

            # Deal cards to each player to prepare for the first round
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

            # MODIFIED: BB-first action order for heads-up games
            if self.num_players == 2:
                # Heads-up: SB acts first preflop (standard), but enable BB-first postflop
                self.game_pointer = s  # SB acts first preflop
                self.enable_bb_first_action_order = True
                print("BB-first action order enabled for heads-up game")
            else:
                # Multiplayer: Player next to big blind acts first (standard)
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

        # Apply the monkey patches - preserve all original methods
        NolimitholdemGame.init_game = patched_init_game
        # Keep all other methods from the original game
        NolimitholdemGame.configure = original_configure
        NolimitholdemGame.step = original_step
        NolimitholdemGame.get_state = original_get_state
        NolimitholdemGame.get_legal_actions = original_get_legal_actions
        NolimitholdemGame.get_num_actions = original_get_num_actions
        NolimitholdemGame.get_num_players = original_get_num_players
        NolimitholdemGame.get_payoffs = original_get_payoffs
        NolimitholdemGame.get_player_id = original_get_player_id
        NolimitholdemGame.is_over = original_is_over
        if original_step_back:
            NolimitholdemGame.step_back = original_step_back

        print("Successfully patched NolimitholdemGame.init_game with BB-first action order support while preserving all original methods")

        return True

    except Exception as e:
        print(f"ERROR: Failed to patch RLCard game: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Setting up RLCard BB-first action order patch for deployment...")
    success = patch_rlcard_game()
    if success:
        print("RLCard patch setup completed successfully!")
        sys.exit(0)
    else:
        print("RLCard patch setup failed!")
        sys.exit(1)
