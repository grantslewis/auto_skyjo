from skyjo_gym_game.game.skyjo_card import SkyjoCard
from skyjo_gym_game.game.utils import cards2list, COL_COUNT, COL_REMOVE_MULT, ENDED_WINNER_MULT, POS_END_CHANGE_MULT, NEG_END_CHANGE_MULT, STEP_CHANGE_MULT, TRAIT_MAP, UPPER_THRESH, ABOVE_THRESH_MULT, BELOW_THRESH_START
import numpy as np

class SkyjoRound:
    
    def __init__(self, dealer, players, np_random):
        self.np_random = np_random
        self.dealer = dealer
        self.players = players
        
        self.current_player_ind = 0
        self.current_player = self.players[self.current_player_ind]
        self.num_players = len(self.players)
        # self.played_cards = []
        self.is_over = False
        self.player_id_that_ended = None
        self.winner = None
        self.final_results = [0 for _ in range(self.num_players)]
        
        self.dealer.deal_cards(self.players)
        self.dealer.played_cards.append(self.dealer.flip_top_card())
        
        self.applied_end_change = [False for _ in range(self.num_players)]
        
        # self.direction = 1
        # self.target = None
    
    # def observe(self):
    #     # return self.get_formatted_state()
    #     pass
    
    def observe(self):
        
        player_amounts = self.current_player.get_visible_amount()
        
        other_player_amounts = []
        for i in self.other_player_inds(self.current_player_ind):
            other_player_amounts.append(self.players[i].get_visible_amount())
            
        
        # if low unknowns and low value, put first
        # NOTE: Might need to change sorting function
        other_player_amounts = np.array(sorted(other_player_amounts, key=lambda x: (x[1] + x[0])))
        
            
            # other_player_amounts.append(self.players[i].get_visible_amount())
        
        
        state = {'player': self.current_player.get_state(),
                 'next_player': self.players[self.next_player_ind()].get_state(),
                 'player_amounts': player_amounts,
                 'other_player_amounts': other_player_amounts,
                 'discard': TRAIT_MAP.index(self.dealer.played_cards[-1]()),
                #  'deck': TRAIT_MAP.index(self.dealer.peak()),
                 'draw': TRAIT_MAP.index(self.dealer.peak()),
                 'legal_flip': self.current_player.get_legal_flip(),
                 'legal_replace': self.current_player.get_legal_replace(),
                 'current_player': self.current_player_ind,
                 }
        # print('field:')
        # self.current_player.print_field()
        
        # print('round', state)
        
        return state
    
    # def get_all_current_scores(self, include_unknowns=False):
    #     scores
        
        
    def next_player_ind(self, curr_player=None):
        if curr_player is None:
            curr_player = self.current_player_ind
        return (curr_player + 1) % self.num_players
    
    def other_player_inds(self, curr_player=None):
        if curr_player is None:
            curr_player = [i for i in range(self.num_players)] #self.current_player_ind
        return [(curr_player + i) % self.num_players for i in range(1, self.num_players)]
    
    def end_game(self):
        self.is_over = True
        self.player_id_that_ended = self.current_player_ind
        remove_count = []
        for i in range(self.num_players):
            self.final_results[i] = self.players[i].get_score(include_unknowns=True)
            remove_count.append(self.players[i].get_remove_count())
        self.na_count = remove_count
        self.final_results_raw = self.final_results.copy()
        
        ender_results = self.final_results[self.player_id_that_ended]
        ender_won = True
        for i in range(self.num_players):
            if self.final_results[i] <= ender_results and i != self.player_id_that_ended:
                ender_won = False
        if not ender_won:
            self.final_results[self.player_id_that_ended] *= 2
        self.winner = np.argmin(self.final_results)
        winner_value = self.final_results[self.winner]
        
        
        max_val, min_val = max(self.final_results), min(self.final_results)
        # new_results = [0 for _ in range(self.num_players)]
        new_results = []
        winners = []
        for i, (val, rm_cnt) in enumerate(zip(self.final_results, remove_count)):
            change = 0
            
            is_winner = (val == self.final_results[self.winner])
            if is_winner:
                winners.append(i)
                
            if val < UPPER_THRESH:
                change += BELOW_THRESH_START / val
                
                if is_winner:
                    amount = (max_val - min_val)
                    amount += 200 #1
                    if i == self.player_id_that_ended and ender_won:
                        # NOTE: Hardcoded
                        # amount += 2 # 10
                        amount *= ENDED_WINNER_MULT
                    
                    change += (amount * POS_END_CHANGE_MULT)
                else:
                    change += ((winner_value - val) * NEG_END_CHANGE_MULT)
                    
            else:
                change += -1 * val * ABOVE_THRESH_MULT
            
            change += (rm_cnt * COL_REMOVE_MULT)
            new_results.append(change)
        self.final_results_actual = self.final_results.copy()
        if len(winners) > 1 or winners[0] != self.player_id_that_ended:
            self.final_results_actual[self.player_id_that_ended] *= 2
        
        self.final_results = new_results
        self.winners = winners
        
        
        # self.final_results = [res - self.final_results[self.winner] + 1 for res in self.final_results]
        # self.final_results = [(self.final_results[i] - self.final_results[self.winner]) + 1 for i in range(self.num_players)]
        return
        
        
    def step(self, action):
        card_action, location_action = action[0], action[1]
        # change = 0
        if not self.is_over:
            row, col = location_action // COL_COUNT, location_action % COL_COUNT
            update_card = None
            if card_action == 0:
                update_card = self.dealer.played_cards.pop()
            elif card_action == 1:
                update_card = self.dealer.flip_top_card()
            #     self.current_player.flip_card(row, col)
            #     self.
            #     return
            change = self.current_player.get_score(include_unknowns=False)
            # discard = self.current_player.update_card(row, col, update_card)
            discard = self.current_player.replace(row, col, update_card)
            
            if discard is None: # card_action == 2
                discard = self.dealer.flip_top_card()
            change = change - self.current_player.get_score(include_unknowns=False)
            change *= STEP_CHANGE_MULT
            self.dealer.played_cards.append(discard)
        
        is_finished = False
        
        if self.current_player.is_end():
            self.end_game()
        
        if self.is_over:
            if self.applied_end_change[self.current_player_ind]:
                change = 0
                is_finished = True
            print(f'game is over, player: {self.current_player_ind},', self.final_results[self.current_player_ind], ' person total:', self.current_player.get_score(include_unknowns=True))
            change = self.final_results[self.current_player_ind]
            self.applied_end_change[self.current_player_ind] = True
            
        
        # elif self.current_player.is_end():
        #     self.end_game()
        #     # change += self.final_results[self.current_player_ind] 
        #     change = self.final_results[self.current_player_ind]
        #     change *= END_CHANGE_MULT if END_CHANGE_MULT is not None else max([abs(res) for res in self.final_results])
          
        print('action: {}, {}, change: {}'.format(card_action, location_action, change))
        self.current_player.print_field(peak=self.is_over)
          
        self.current_player_ind = self.next_player_ind()
        self.current_player = self.players[self.current_player_ind]
        # is_finished = self.is_over
        if self.player_id_that_ended is not None and self.current_player_ind == self.player_id_that_ended:
            is_finished = True
            
        
        # if self.current_player.is_human():
        
        
        obs = self.observe()
        # print('step obs:', obs)
        # print('step obs:', obs)
        return obs, change, is_finished, {}#, self.current_player.is_human()
    
    
    def print_borad(self):
        curr_player = self.current_player_ind
        for _ in range(self.num_players):
            print('Player {}:'.format(curr_player.get_player_id()))
            self.players[curr_player].print_board()
            curr_player = self.next_player_ind(curr_player)
            print('')
        return
                 
                     
    
    # def flip_top_card(self):
    #     top = self.dealer.flip_top_card()
    #     self.target = top
    #     self.dealer.played_cards.append(top)
    #     return top

    # def proceed_round(self, players, action):
    #     if action == 'draw':
    #         self._perform_draw_action(players)
    #         return None
    #     player = players[self.current_player_ind]
    #     # card_info = action.split('-')
        
        
        
    # def get_legal_actions(self, players, player_id=None):
    #     if player_id is None:
    #         player_id = self.current_player_ind
    #     return players[player_id].get_legal_actions()
    
    # def get_state(self, players, player_id=None):
    #     curr_player = self.current_player if player_id is None else self.players[player_id]
    #     # if player_id is None:
    #         # player_id = self.current_player_ind
    #     next_player = self.players[self.next_player_ind(player_id)]
    #     ret_state = dict()
    #     # next_player_id = (player_id + 1) % self.num_players
    #     # next_player_id = self.next_player_ind(player_id)
        
    #     ret_state['discard'] = self.dealer.played_cards[-1]()
    #     ret_state['next_card'] = self.dealer.deck[-1]()
    #     ret_state['player'] = curr_player.get_state()
    #     ret_state['next_player'] = next_player.get_state()
    #     ret_state['legal_actions'] = curr_player.get_legal_actions()
        
    #     # ret_state['player'] = players[player_id].get_state()
    #     # ret_state['next_player'] = players[next_player_id].get_state()
    #     # ret_state['legal_actions'] = players[player_id].get_legal_actions()
                
    #     return ret_state
    
    # replace_deck not needed
             
    # def _perform_draw_action(self, players):
    #     # players[self.current_player_ind].add_card(self.dealer.deck.pop())
    #     # self.current_player_ind = (self.current_player_ind + 1) % self.num_players
    #     pass
    
    # _perform_non_number_action not needed
    

    # tqdm(data, description=)