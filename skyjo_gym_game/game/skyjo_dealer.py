from skyjo_gym_game.game.utils import DECK_AMOUNTS, FIELD_SIZE
from skyjo_gym_game.game.skyjo_card import SkyjoCard as Card
import numpy as np

class SkyjoDealer:
    ''' Initialize a uno dealer class
    '''
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = SkyjoDealer._init_deck()
        self.played_cards = []
        # self.deck = SkyjoDealer.shuffle(self.deck)

    

    def deal_cards(self, players, num=FIELD_SIZE): #, num, sort_field=False):
        ''' Deal some cards from deck to each player
        Args:
            players (list of objects): The object of SkyjoPlayer
            num (int): The number of cards to be dealt
        '''
        # if not isinstance(players, list):
        #     players = [players]
        
        for player in players:
            for _ in range(num):
                # player.cards.append(self.deck.pop())
                player.add_card(self.deck.pop())
            player.starting_flip()
            
            
        
        # print('post dealing')        
        # players[0].print_field()
        # print('peak')
        # players[0].print_field(peak=True)
        
                
        # if sort_field:
        # for player in players:
        #     player.sort_field()
    def deck_check(self, only_if_empty=True):
        if not only_if_empty or len(self.deck) == 0:
        
            last_card = self.played_cards.pop()
            print(self.played_cards)
            cards = self.played_cards.copy()
            for card in cards:
                card.reset()
            # cards = [card.reset() for card in self.played_cards]
            # print(cards)
            deck = cards + self.deck
            # print(deck)
            deck = SkyjoDealer.shuffle(deck)
            
            self.deck = deck
            self.played_cards = [last_card]
            print('deck_len', len(self.deck), 'played_cards_len', len(self.played_cards))
            print('deck', self.deck)
    
        
    def flip_top_card(self):
        ''' Flip top card when a new game starts
        Returns:
            (object): The object of UnoCard at the top of the deck
        '''
        self.deck_check()
        top_card = self.deck.pop()
        top_card.flip()
        return top_card

    def peak(self):
        self.deck_check()
        return self.deck[-1].value
    
    @staticmethod
    def shuffle(deck):
        ''' Shuffle the deck
        '''
        deck = np.array(deck.copy())
        np.random.shuffle(deck)
        return deck.tolist()
    
    @staticmethod
    def _init_deck():
        deck = []
        for key, value in DECK_AMOUNTS.items():
            # deck.extend([key]*value)
            deck.extend([Card(key) for _ in range(value)] )
        
        deck = SkyjoDealer.shuffle(deck)
                
        # deck = np.array(deck)
        # np.random.shuffle(deck)
        # deck = deck.tolist()
        
        # Card.print_cards(deck[:48])
        return deck
    
    
    