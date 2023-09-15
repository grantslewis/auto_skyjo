from skyjo_gym_game.game.skyjo_card import SkyjoCard
from skyjo_gym_game.game.utils import cards2list
import numpy as np

class SkyjoRound:
    
    def __init__(self, dealer, num_players, np_random):
        self.np_random = np_random
        self.dealer = dealer
        self.target = None
        self.current_player = 0
        self.num_players = num_players
        # self.direction = 1
        self.played_cards = []
        self.is_over = False
        self.winner = None
    
    def flip_top_card(self):
        top = self.dealer.flip_top_card()
        self.target = top
        self.played_cards.append(top)
        return top

    def proceed_round(self, players, action):
        if action == 'draw':
            self._perform_draw_action(players)
            return None
        player = players[self.current_player]
        # card_info = action.split('-')
        
        
        
    def get_legal_actions(self, players, player_id):
        return players[player_id].get_legal_actions()
    
    def get_state(self, players, player_id):
        ret_state = dict()
        next_player_id = (player_id + 1) % self.num_players
        
        ret_state['discard'] = self.played_cards[-1].value
        ret_state['next_card'] = self.dealer.deck[-1].value
        ret_state['player'] = players[player_id].get_state()
        ret_state['next_player'] = players[next_player_id].get_state()
        ret_state['legal_actions'] = players[player_id].get_legal_actions()
                
        return ret_state
    
    # replace_deck not needed
             
    def _perform_draw_action(self, players):
        # players[self.current_player].add_card(self.dealer.deck.pop())
        # self.current_player = (self.current_player + 1) % self.num_players
        pass
    
    # _perform_non_number_action not needed