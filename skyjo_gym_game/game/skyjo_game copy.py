from copy import deepcopy
import numpy as np

from skyjo_gym_game.game.skyjo_dealer import SkyjoDealer as Dealer
from skyjo_gym_game.game.skyjo_player import SkyjoPlayer as Player
from skyjo_gym_game.game.skyjo_round import SkyjoRound as Round
from skyjo_gym_game.game.utils import ROW_COUNT, COL_COUNT, NA_VAL, UNK_VAL, NUM_ACTIONS


class SkyjoGame:
    
    def __init__(self, allow_step_back=False, num_players=2):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = np.random.RandomState()
        self.payoffs = [0 for _ in range(self.num_players)]
        self.player_that_ended = None

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']
    
    def init_game(self):
        ''' Initialize players and state
        Returns:
            (tuple): Tuple containing:
                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize payoffs
        self.payoffs = [0 for _ in range(self.num_players)]
        
        # init dealer
        self.dealer = Dealer(self.np_random)
        
        # init players
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]
        
        # Deal cards to players
        self.dealer.deal_cards(self.players, (ROW_COUNT * COL_COUNT))
        
        # init a round
        self.round = Round(self.dealer, self.players, self.np_random)
        
        self.round.flip_top_card()
        
        # Save the hisory for stepping back to the last state.
        self.history = []
        
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id
    
    def step(self, action):
        ''' Get the next state
        Args:
            action (str): A specific action
        Returns:
            (tuple): Tuple containing:
                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id
        
    def step_back(self):
        ''' Return to the previous state of the game
        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        ''' Return player's state
        Args:
            player_id (int): player id
        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        state['num_players'] = self.num_players #self.get_num_players()
        state['current_player'] = self.round.current_player
        return state
        
    def get_payoffs(self):
        ''' Return the payoffs of the game
        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        
        scores = [player.get_score(True) for player in self.players]
        min_val, max_val = min(scores), max(scores)
        for i, score in enumerate(scores):
            if i != self.player_that_ended and score == min_val:
                score[i] = score[i] * 2
        
        inverted = [1 / (score - min_val + 1) for score in scores] # from 0 - inf  to 1 - 0
        return inverted
        # winner = self.round.get_winner(self.players)
        # pass

    def get_legal_actions(self):
        ''' Return the legal actions for current player
        Returns:
            (list): A list of legal actions
        '''

        return self.round.get_legal_actions(self.players, self.round.current_player)

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions
        Returns:
            (int): The number of actions. There are 61 actions
        '''
        return NUM_ACTIONS

    def get_player_id(self):
        ''' Return the current player's id
        Returns:
            (int): current player's id
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over
        Returns:
            (boolean): True if the game is over
        '''
        return self.round.is_over