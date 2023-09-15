from copy import deepcopy
import numpy as np

from skyjo_gym_game.game.skyjo_dealer import SkyjoDealer as Dealer
from skyjo_gym_game.game.skyjo_player import SkyjoPlayer as Player
from skyjo_gym_game.game.skyjo_round import SkyjoRound as Round
from skyjo_gym_game.game.utils import ROW_COUNT, COL_COUNT, NA_VAL, UNK_VAL, NUM_ACTIONS


class SkyjoGame:
    def __init__(self, num_players=2,  allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = np.random.RandomState()
        self.payoffs = [0 for _ in range(self.num_players)]
        self.player_that_ended = None
        players = [Player(i, self.np_random) for i in range(self.num_players)]
        self.round = Round(Dealer(self.np_random), players, self.np_random)
        # print('game', self.round)
        # self.round = Round(Dealer(self.np_random), players, self.np_random)
    
    # def reset(self):
    #     del self.round
        
    def observe(self):
        # print('game obs')
        obs = self.round.observe()
        # print(obs)
        return obs
    
    def step(self, action):
        # print('game step', action)
        obs, reward, done, other = self.round.step(action)
        
        # obs = self.round.step(action[0], action[1])
        
        return obs, reward, done, other
        # convert to 
        pass
    
    
        
    def init_game(self):
        
        pass

    # def action(self, action):
    #     # convert to 
    #     pass

    def evaluate(self):
        pass

    def is_done(self):
        return self.round.is_over

    # def observe(self):
    #     pass

    def view(self):
        pass
    
    def current_player(self):
        return 0
    
    def next_player(self):
        return (self.current_player() + 1) % self.num_players