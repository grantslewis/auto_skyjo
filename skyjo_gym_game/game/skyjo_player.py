import numpy as np
from skyjo_gym_game.game.utils import COL_COUNT, ROW_COUNT, NA_VAL, NA_INT, UNK_VAL, UNK_INT, TRAIT_MAP, START_FLIP
from skyjo_gym_game.game.skyjo_card import SkyjoCard as Card
import random

class SkyjoPlayer:
    
    def __init__(self, player_id, np_random, is_human_player=False):
        ''' Initilize a player.
        Args:
            player_id (int): The id of the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.field = []
        self.is_human_player = is_human_player
    
    def is_human(self):
        return self.is_human_player
    
    def starting_flip(self, flip_count=START_FLIP, is_random=False):
        ''' Flip some cards at the start of the game
        Args:
            flip_count (int): The number of cards to be flipped
        '''
        # get n random indicies
        # inds = self.np_random.choice((ROW_COUNT * COL_COUNT), flip_count, replace=False)
        
        # get last n indicies
        inds = []
        if is_random:
            inds = random.sample(range(ROW_COUNT * COL_COUNT), flip_count)
        else:
            inds = np.arange(ROW_COUNT * COL_COUNT)[-flip_count:]
        
        for i in range(flip_count):
            col = inds[i] // ROW_COUNT
            row = inds[i] % ROW_COUNT
            self.flip(row, col)
            
            # row = self.np_random.randint(ROW_COUNT)
            # col = self.np_random.randint(COL_COUNT)
            # self.flip(row, col)
        
    def add_card(self, card):
        self.field.append(card)
        
        if len(self.field) == (ROW_COUNT * COL_COUNT):
            self.field = np.reshape(self.field, (ROW_COUNT, COL_COUNT))
        return
    
    def update_column(self, col):
        print('updating column')
        self.print_field()
        val = None
        for card in self.field[:, col]:
            if card.trait == UNK_VAL:
                return
            if val is None:
                val = card.trait
            elif val != card.trait:
                return
        for card in self.field[:, col]:
            card.remove(is_col=True)

    def flip(self, row, col):
        print('flip', row, col)
        self.field[row][col].flip()
        self.update_column(col)
    
    def replace(self, row, col, card):
        # discard = self.field[row][col].flip() if card is not None else None
        discard = None
        if card is None:
            self.flip(row, col)
        else:
            self.field[row][col].flip()
            discard = self.field[row][col]
            self.field[row][col] = card
            self.update_column(col)
        print(f'player: {self.player_id} replace; original:', self.field[row][col].value, row, col, 'new:', card() if card is not None else None, 'discard:', discard() if discard is not None else None)
        return discard

    def get_score(self, include_unknowns=True):
        score_ls = [int(card.value) for card in self.field.flatten() if (card() != NA_VAL and (card() != UNK_VAL or include_unknowns))]
        score = sum(score_ls)
        # print(score, score_ls)
        return score
        # score = 0
        # for card in self.field.flatten():
        #     # if (card() == UNK_VAL and include_unknowns) or (card() != NA_VAL and card() != UNK_VAL):
        #     if (card() != NA_VAL and (card() != UNK_VAL or include_unknowns)):
        #         score += int(card.value)
            
        #     # if card.trait == UNK_VAL and include_unknowns:
        #     #     # if include_unknowns:
        #     #     score += card.value
        #     # elif card.trait != NA_VAL:
        #     #     score += card.trait
        # return score

    def get_remove_count(self):
        
        return len([card for card in self.field.flatten() if card() == NA_VAL])
        
            
    
    def get_state(self):
        # state = self.field.flatten()
        # state = [TRAIT_MAP[v] for v in state]
        
        # state = self.field
        # state = np.array([[TRAIT_MAP.index(card()) for card in row] for row in state])
        state = np.array([[TRAIT_MAP.index(card()) for card in row] for row in self.field])        
        
        return state
    
    def get_visible_amount(self):
        
        
        # state = self.get_state()
        value_count = 0
        unknown_count = 0
        for row in self.field:
            for card in row:
                c_val = card()
                # print(c_val)
                if c_val == UNK_VAL: #UNK_INT:
                    unknown_count += 1
                elif c_val != NA_VAL: # NA_INT:
                    value_count += int(c_val)
        return [value_count, unknown_count]                  
    
    def get_legal_flip(self):
        legals = np.array([[1 if card() == UNK_VAL else 0 for card in row] for row in self.field])
        return legals
    
    def get_legal_replace(self):
        legals = np.array([[1 if card() != NA_VAL else 0 for card in row] for row in self.field])
        return legals

    # def get_legal_actions(self):
    #     states = self.field #.flatten()
    #     states = np.where(states != NA_VAL, 1, 0)
    #     # states = [1 if v != NA_VAL else 0 for v in states]
    #     states = np.array([states, states])
    #     return states
    
    def is_end(self):
        for row in self.field:
            for card in row:
                if card.trait == UNK_VAL:
                    return False
        return True
    
    def get_player_id(self):
        ''' Return the id of the player
        '''

        return self.player_id
    
    def print_field(self, peak=False):
        field = self.field.flatten()
        if peak:
            field = [v.value if v.trait != NA_VAL else v.trait for v in field]
        Card.print_cards(field)