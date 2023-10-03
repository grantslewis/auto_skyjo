from termcolor import colored
from skyjo_gym_game.game.utils import CARD_COLORS, COL_COUNT, ROW_COUNT, NA_VAL, UNK_VAL
import numpy as np

# In Utils:
# CARD_COLORS = {-2:"blue", -1:"blue",
#             0:"cyan",
#             1: "green", 2:"green", 3:"green", 4:"green",
#             5:"yellow", 6:"yellow", 7:"yellow", 8:"yellow",
#             9:"red", 10:"red", 11:"red", 12:"red",
#             NA_VAL:"black",
#             UNK_VAL:"white"}


class SkyjoCard:
    def __init__(self, value) -> None:
        self.value = str(value)
        self.trait = UNK_VAL
    
    def __call__(self):
        return self.trait
        
    def flip(self):
        if self.trait != NA_VAL:
            self.trait = self.value
    
    def reset(self):
        self.trait = UNK_VAL
    
    def remove(self, is_col=False):
        print('REMOVING VAL!!!')
        self.trait = NA_VAL
        if not is_col:
            raise Exception('failed')
        
    def get_str(self):
        return str(self.trait)
    
    @staticmethod
    def print_cards(cards, fancy_print=True, print_array=True, return_only=False):
        ''' Print out card in a nice form
        Args:
            card (str, int, or list): The string form or a list of a UNO card
        '''
        if isinstance(cards, (int,str)):
            cards = [str(cards)]
        
        output = []
        cards = np.array(cards).flatten()
        
        if isinstance(cards[0], SkyjoCard):
            cards = np.array([card() for card in cards])
            
        if print_array and len(cards) == (ROW_COUNT * COL_COUNT):
            cards = np.reshape(cards, (ROW_COUNT, COL_COUNT))
            
            for row in cards:
                row_out = []
                for card in row:
                    if fancy_print:
                        row_out.append(colored(card, CARD_COLORS[card]))
                    else:
                        row_out.append(f'{card}')
                output.append(" ".join(row_out))
        else:
            row_out = []
            for card in cards:
                if fancy_print:
                    row_out.append(colored(card, CARD_COLORS[card]))
                else:
                    row_out.append(f'{card}')
            output.append(", ".join(row_out))
        res = "\n".join(output)
        if not return_only:
            print(res)
        return res
    
    
    # VERSION 2 started
    # @staticmethod
    # def print_cards(cards, fancy_print=True, print_array=True, return_only=False):
    #     ''' Print out card in a nice form
    #     If it is not return_only, then it will print out the cards
    #     Args:
    #         card (str, int, list, or list of lists): will convert all to list of lists, where each sublist is a player
            
    #     '''
    #     if isinstance(cards, (int,str)):
    #         cards = [str(cards)]
    #     if not isinstance(cards[0], list):
    #         cards = [cards]
            
        
    #     cards = [np.array(card_group).flatten() for card_group in cards]
        
    #     output = []
        
    #     if isinstance(cards[0][0], SkyjoCard):
    #         cards = np.array([[card() for card in card_group] for card_group in cards])
        
        
            
        
        
    #     if print_array and len(cards) == (ROW_COUNT * COL_COUNT):
    #         cards = np.reshape(cards, (ROW_COUNT, COL_COUNT))
            
    #         for row in cards:
    #             row_out = []
    #             for card in row:
    #                 if fancy_print:
    #                     row_out.append(colored(card, CARD_COLORS[card]))
    #                 else:
    #                     row_out.append(f'{card}')
    #             output.append(" ".join(row_out))
    #     else:
    #         row_out = []
    #         for card in cards:
    #             if fancy_print:
    #                 row_out.append(colored(card, CARD_COLORS[card]))
    #             else:
    #                 row_out.append(f'{card}')
    #         output.append(", ".join(row_out))
    #     res = "\n".join(output)
    #     if not return_only:
    #         print(res)
    #     return res
