import numpy as np

UNK_VAL = '?'
NA_VAL = '-'

# a map of trait to its index
# TRAIT_MAP = {'-2': 0, '-1': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7,
#              '6': 8, '7': 9, '8': 10, '9': 11, '10': 12,
#              '11': 13, '12': 14, UNK_VAL: 15, NA_VAL: 16}
# TRAIT_MAP = ['-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', UNK_VAL, NA_VAL]

TRAIT_MAP = ['-2', '-1', '0', NA_VAL, '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', UNK_VAL]


# NOTE: HARD CODED
MAX_COUNT = (12 * 10) + (11 * 2)
MIN_COUNT = (-2 * 5) + (-1 * 7)
TOTAL_COUNT_VARIANCE = MAX_COUNT - MIN_COUNT

UNK_INT = TRAIT_MAP.index(UNK_VAL)
NA_INT = TRAIT_MAP.index(NA_VAL)
print('unk and na:', UNK_INT, NA_INT)


STEP_CHANGE_MULT = 1 / 14
ENDED_WINNER_MULT = 2 #3 #6 #12 #100
POS_END_CHANGE_MULT = 2 #2.5 #5 #10 #100 # None if should be max player score
NEG_END_CHANGE_MULT = 1 #5 #2 #100 # None if should be min player score
COL_REMOVE_MULT = 2.5 #5 #10 # NA Mult

UPPER_THRESH = 30
ABOVE_THRESH_MULT = 2
BELOW_THRESH_START = 2000 #UPPER_THRESH

COL_COUNT = 4
ROW_COUNT = 3
FIELD_SIZE = COL_COUNT * ROW_COUNT
# NUM_ACTIONS = ((COL_COUNT * ROW_COUNT) * 2) + 1
NUM_ACTIONS = 12

START_FLIP = 2

MAX_PLAYER_COUNT = 8

CARD_COLORS = {"-2":"blue", "-1":"blue",
            "0":"cyan",
            "1": "green", "2":"green", "3":"green", "4":"green",
            "5":"yellow", "6":"yellow", "7":"yellow", "8":"yellow",
            "9":"red", "10":"red", "11":"red", "12":"red",
            UNK_VAL:"white", NA_VAL:"black"}

DECK_AMOUNTS = {-2:5, -1:10, 0:15, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:10, 12:10}

def cards2list(cards):
    ''' Get the corresponding string representation of cards
    Args:
        cards (list): list of UnoCards objects
    Returns:
        (string): string representation of cards
    '''
    cards = np.array(cards)
    cards_list = []
    for card in cards:
        cards_list.append(card.get_str())
    return cards_list

def ind_to_row_col(ind):
    return (ind // COL_COUNT, ind % COL_COUNT)