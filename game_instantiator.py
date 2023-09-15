import gym
from agent import Agent, DQNAgent
import skyjo_gym_game
import numpy as np


num_players = 6 # 2
DRAW_HIDDEN = (24, 32, 24)
FLIP_HIDDEN = (24, 32, 24)
LOCATION_HIDDEN = (128, 256, 256, 128)
field_size = 12



INPUT_OPTIONS = {'player': field_size,
                'next_player': field_size, # * (num_players - 1),
                'discard': 1,
                'draw':1,
                'player_amounts': 2,
                'other_player_amounts': 2 * (num_players - 1)}
DRAW_COLS = ['player', 'discard', 'player_amounts', 'other_player_amounts']
FLIP_COLS = ['player', 'draw', 'player_amounts', 'other_player_amounts']
LOCATION_COLS_D = ['player', 'next_player', 'discard', 'player_amounts', 'other_player_amounts']
LOCATION_COLS_F = ['player', 'next_player', 'draw', 'player_amounts', 'other_player_amounts']

INPUT_SIZE = sum(INPUT_OPTIONS.values())
state_selection = INPUT_OPTIONS.keys()

def instantiate(learning_rate, discount_factor, replay_buffer_size, batch_size):

    env = gym.make('skyjo-v0', num_agents=num_players)

    draw_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in DRAW_COLS]), 2, DRAW_HIDDEN, 
                            discount_factor, learning_rate, 
                            replay_buffer_size, batch_size) for _ in range(num_players)]
    flip_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in FLIP_COLS]), 2, FLIP_HIDDEN, 
                            discount_factor, learning_rate, 
                            replay_buffer_size, batch_size) for _ in range(num_players)]
    location_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in LOCATION_COLS_D]), field_size, LOCATION_HIDDEN, 
                            discount_factor, learning_rate, 
                            replay_buffer_size, batch_size) for _ in range(num_players)]
    return num_players, env, draw_agent, flip_agent, location_agent


    # epsilons, draw_agent, flip_agent, location_agent = load_previous(num_players, epsilons, draw_agent, flip_agent, location_agent)

# def calc_action(state, agent, epsilon, mask=None):
#     res = agent(state, epsilon)
#     # print(res)
#     # res = res.detach().numpy()
#     if mask is not None:
#         res = res * np.array(mask).flatten()
#     return np.argmax(res)

def calc_action(state, agent, epsilon, mask=None):
    res = agent(state, epsilon)
    # print(res)
    # res = res.detach().numpy()
    if mask is not None and not all(val==1 for val in mask):
        # print('was masked')
        # print(res, mask)
        res = res * np.array(mask).flatten()
        # exit(0)
        if max(res) <= 0:
            reset_zeros = (min(res) // 1) - 1
            update_zeros = [reset_zeros if i == 0 else 0 for i in mask]
            res = [old + update for (old, update) in zip(res, update_zeros)]
            
            # print(res)
            # raise Exception('res is too small')
    return np.argmax(res)
    
class ActionInfo:
    def __init__(self, state, cols_to_sel, can_be_missing=False, is_run=True) -> None:
        self.state_orig = state
        self.cols_to_sel = cols_to_sel
        for col in cols_to_sel:
            if col not in state and not can_be_missing:
                raise Exception(f'col: {col} not in state: {state}')
        self.state = None
        
        # state_sel = [state[col] for col in cols_to_sel]
        # if len (state_sel) != len(cols_to_sel) and not can_be_missing:
        #     raise Exception(f'cols_to_sel: {cols_to_sel} not all in selected state: {state_sel}')
        
        # self.state = state_sel
        
        # self.main_col = main_col
        # self.append_col = append_col
        # self.state = state[main_col]
        # self.state.append(state[append_col])
        self.set_state(state, is_next=False)
        
        self.action = 0
        self.reward = 0
        self.done = False
        self.next_state = None
        self.is_run = is_run
    
    def set_outputs(self, next_state, reward, done):
        self.set_state(next_state)
        self.reward = reward
        self.done = done
        
    def set_state(self, state, is_next=True, append_col=None):
        sel_state = []
        for col in self.cols_to_sel:
            curr_sel = state[col]
            if isinstance(curr_sel, np.ndarray):
                # print('is_ndarray')
                curr_sel = state[col].flatten().tolist()
            if not isinstance(curr_sel, list):
                curr_sel = [curr_sel]
            sel_state.extend(curr_sel)
            
            # if isinstance(state[col], int):
            #     sel_state.append(state[col])
            # else:
            #     sel_state.extend()
        
        
        # sel_state = state[self.main_col].flatten().tolist()
        # append_col = self.append_col if append_col is None else append_col
        # sel_state.append(state[append_col])
        if is_next:
            self.next_state = sel_state
        else:
            self.state = sel_state
        # print('sel_state', sel_state)
    
    def get_tuple(self):
        if not self.is_run:
            return None
        return (self.state, self.action, self.reward, self.next_state, self.done)
    
    def calc_action(self, agent, epsilon, mask=None):
        self.is_run = True
        self.action = calc_action(self.state, agent, epsilon, mask)
        return self.action

