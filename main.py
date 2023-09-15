import gym
# import gym_game
import skyjo_gym_game
import numpy as np
import random
from agent import Agent, DQNAgent
import PIL
import os
import glob
import time
import json
import matplotlib.pyplot as plt
from game_instantiator import instantiate, ActionInfo, DRAW_COLS, FLIP_COLS, LOCATION_COLS_D, LOCATION_COLS_F

field_size = 12
num_vals = 17

TRAIN_ROUND = 0
OVERRIDE_PREV = False #True

# NUM_PLAYERS = 6 # 8 # CHANGE GAME INSTANTIATOR

# PARENT_FOLDER = './agent_results_3'
# PARENT_FOLDER = './agent_results_8players'
PARENT_FOLDER = './agent_results_6players'


# DRAW_HIDDEN = (24, 32, 24)
# FLIP_HIDDEN = (24, 32, 24)
# LOCATION_HIDDEN = (128, 256, 256, 128)


if not OVERRIDE_PREV:
    while os.path.exists(f'{PARENT_FOLDER}/draw_agent/{TRAIN_ROUND}') and len(os.listdir(f'{PARENT_FOLDER}/draw_agent/{TRAIN_ROUND}')) > 0:
        TRAIN_ROUND += 1

DRAW_PATH = f'{PARENT_FOLDER}/draw_agent/{TRAIN_ROUND}'
FLIP_PATH = f'{PARENT_FOLDER}/flip_agent/{TRAIN_ROUND}'
LOCATION_PATH = f'{PARENT_FOLDER}/location_agent/{TRAIN_ROUND}'
RESULTS_PATH = f'{PARENT_FOLDER}/results/'

PREV_ROUND = None
PREV_ROUND = TRAIN_ROUND - 1 if PREV_ROUND is None else PREV_ROUND
PREV_DRAW_PATH = f'{PARENT_FOLDER}/draw_agent/{PREV_ROUND}'
PREV_FLIP_PATH = f'{PARENT_FOLDER}/flip_agent/{PREV_ROUND}'
PREV_LOCATION_PATH = f'{PARENT_FOLDER}/location_agent/{PREV_ROUND}'
NUM_TIMES_TO_LOAD_BEST = 1 #2


UPDATE_FREQUENCY = 50

SAVE_FREQUENCY = 250

# MAX_EPISODES = 5000 #1500 # 50 #1000
MAX_EPISODES = 10000
# MAX_EPISODES = 250
MAX_TRY = 500

WARM_UP_TIME = 100

MAX_REWARD_FOR_TIME = -3
TIME_REWARD_MULT = MAX_REWARD_FOR_TIME / (MAX_TRY - WARM_UP_TIME)

NOT_FINISHING_REWARD = -100

EPSILON_DECAY_FREQUENCY = 1

MAX_EPSILON_RESET = 0.9


# CHANGES:
# Ensure that the 3 column cards are discarded to properly (with the 4th card if applicable)
# Flip over two cards randomly at the beginning of the game
# modify the input to incorporate other players some how (either their field or a known total and how many cards they have flipped over)
#  Need to also account for differnet number of players 
# Maybe increase reward for getting rid of a column
# Maybe increase reward a ton if ends game with lowest and decrease penalty for ending the game without the lowest
# decrease the penalty of losing?
# allow for random sampling from model output distribution
# Maybe dont give a negative penalty for flipping?? or at least determine based on card chose not to use



# def proceed_action(state, q_table, epsilon):
#     if random.uniform(0, 1) < epsilon:
#         action = env.sample_draw()
#     else:
#         action = np.argmax(q_table[state])
#     return action

# def location_action(state, q_table, epsilon):
#     if random.uniform(0, 1) < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(q_table[state])
#     return action
        
def update_q_table(state, action, reward, next_state, q_table, learning_rate, gamma):
    q_table[state, action] = q_table[state, action] + (learning_rate * (reward + gamma * np.max(q_table[next_state])) - q_table[state, action])
    return q_table

def state_to_one_hot(state):
    if isinstance(state, np.ndarray):
        state = state.flatten().tolist()
    if not isinstance(state, list):
        state = [state]
    one_hot_state = np.zeros((field_size, num_vals)) # num_vals + 1
    one_hot_state[np.arange(field_size), state] = 1
    # one_hot_state[-1, state[2]] = 1
    return one_hot_state

# def calc_action(state, agent, epsilon, mask=None):
#     res = agent(state, epsilon)
#     # print(res)
#     # res = res.detach().numpy()
#     if mask is not None:
#         print('was masked')
#         print(res, mask)
#         res = res * np.array(mask).flatten()
#         exit(0)
#     if max(res) <= 0:
#         print(res)
#         raise Exception('res is too small')
#     return np.argmax(res)




# class ActionInfo:
#     def __init__(self, state, main_col, append_col, is_run=True) -> None:
#         self.main_col = main_col
#         self.append_col = append_col
#         # self.state = state[main_col]
#         # self.state.append(state[append_col])
#         self.set_state(state, is_next=False)
#         self.action = 0
#         self.reward = 0
#         self.done = False
#         self.next_state = None
#         self.is_run = is_run
    
#     def set_outputs(self, next_state, reward, done):
#         self.set_state(next_state)
#         self.reward = reward
#         self.done = done
        
#     def set_state(self, state, is_next=True, append_col=None):
#         sel_state = state[self.main_col].flatten().tolist()
#         append_col = self.append_col if append_col is None else append_col
#         sel_state.append(state[append_col])
#         if is_next:
#             self.next_state = sel_state
#         else:
#             self.state = sel_state
    
#     def get_tuple(self):
#         if not self.is_run:
#             return None
#         return (self.state, self.action, self.reward, self.next_state, self.done)
    
#     def calc_action(self, agent, epsilon, mask=None):
#         self.is_run = True
#         self.action = calc_action(self.state, agent, epsilon, mask)
#         return self.action

# class ActionInfo:
#     def __init__(self, state, cols_to_sel, can_be_missing=False, is_run=True) -> None:
#         self.state_orig = state
#         self.cols_to_sel = cols_to_sel
#         for col in cols_to_sel:
#             if col not in state and not can_be_missing:
#                 raise Exception(f'col: {col} not in state: {state}')
#         self.state = None
        
#         # state_sel = [state[col] for col in cols_to_sel]
#         # if len (state_sel) != len(cols_to_sel) and not can_be_missing:
#         #     raise Exception(f'cols_to_sel: {cols_to_sel} not all in selected state: {state_sel}')
        
#         # self.state = state_sel
        
#         # self.main_col = main_col
#         # self.append_col = append_col
#         # self.state = state[main_col]
#         # self.state.append(state[append_col])
#         self.set_state(state, is_next=False)
        
#         self.action = 0
#         self.reward = 0
#         self.done = False
#         self.next_state = None
#         self.is_run = is_run
    
#     def set_outputs(self, next_state, reward, done):
#         self.set_state(next_state)
#         self.reward = reward
#         self.done = done
        
#     def set_state(self, state, is_next=True, append_col=None):
#         sel_state = []
#         for col in self.cols_to_sel:
#             curr_sel = state[col]
#             if isinstance(curr_sel, np.ndarray):
#                 # print('is_ndarray')
#                 curr_sel = state[col].flatten().tolist()
#             if not isinstance(curr_sel, list):
#                 curr_sel = [curr_sel]
#             sel_state.extend(curr_sel)
            
#             # if isinstance(state[col], int):
#             #     sel_state.append(state[col])
#             # else:
#             #     sel_state.extend()
        
        
#         # sel_state = state[self.main_col].flatten().tolist()
#         # append_col = self.append_col if append_col is None else append_col
#         # sel_state.append(state[append_col])
#         if is_next:
#             self.next_state = sel_state
#         else:
#             self.state = sel_state
#         # print('sel_state', sel_state)
    
#     def get_tuple(self):
#         if not self.is_run:
#             return None
#         return (self.state, self.action, self.reward, self.next_state, self.done)
    
#     def calc_action(self, agent, epsilon, mask=None):
#         self.is_run = True
#         self.action = calc_action(self.state, agent, epsilon, mask)
#         return self.action


def save_process(total_rewards, epsilons, num_players, draw_agent, flip_agent, location_agent, player_stats):
    summed_rewards = np.sum(total_rewards, axis=1)
    order = np.argsort(summed_rewards)[::-1].tolist()
    
    print('total_rewards:', total_rewards)
    print('winner agent:', order) #winner_agent)
    # print('rewards cnt', np.sum(total_rewards, axis=1).shape)
    print('total_rewards:', summed_rewards)
        # PIL.Image.fromarray(env.render())
    draw_losses = [agent.get_losses() for agent in draw_agent]
    avg_draw_losses = [float(np.nanmean(loss)) for loss in draw_losses]
    flip_losses = [agent.get_losses() for agent in flip_agent]
    avg_flip_losses = [float(np.nanmean(loss)) for loss in flip_losses]
    location_losses = [agent.get_losses() for agent in location_agent]
    avg_location_losses = [float(np.nanmean(loss)) for loss in location_losses]
    print('avg draw losses:', avg_draw_losses)
    print('avg flip losses:', avg_flip_losses)
    print('avg location losses:', avg_location_losses)
    print('epsilons:', epsilons)
    # print(draw_losses.shape)
    
        
        
    for i in range(num_players):
        name_modder = ''
        if i == order[0]:
            name_modder = '_winner'
        elif i == order[-1]:
            name_modder = '_loser'
        order_ind = order.index(i)
        draw_agent[i].save_model(os.path.join(DRAW_PATH, f'{order_ind}_{i}{name_modder}.h5'))
        flip_agent[i].save_model(os.path.join(FLIP_PATH, f'{order_ind}_{i}{name_modder}.h5'))
        location_agent[i].save_model(os.path.join(LOCATION_PATH, f'{order_ind}_{i}{name_modder}.h5'))
        
        
    res = {'order': order, 'summed_rewards': summed_rewards.tolist(), 'epsilons': epsilons,
           'avg_draw_losses': avg_draw_losses, 'avg_flip_losses': avg_flip_losses, 'avg_location_losses': avg_location_losses,
           'total_rewards': total_rewards,
           'draw_losses': draw_losses, 'flip_losses': flip_losses, 'location_losses': location_losses,
           }
    
    with open(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.json'), 'w') as f:
        f.write(json.dumps(res))
    
    updated_player_stats = dict() #{k: v for k,v in player_stats.items()}
    for k,v in player_stats.items():
        v_updated = v.copy()
        v_updated['total_reward'] = sum(v['final_results'])
        updated_player_stats[k] = v_updated
    
    
    with open(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}_player_stats.json'), 'w') as f:
        f.write(json.dumps(updated_player_stats)) #player_stats))
    
    
def load_previous(num_players, epsilons, draw_agent, flip_agent, location_agent):
    if os.path.exists(PREV_DRAW_PATH):
        print('loading prev models')
        # time.sleep(1)
        draw_h5s = glob.glob(PREV_DRAW_PATH + '/*.h5')
        flip_h5s = glob.glob(PREV_FLIP_PATH + '/*.h5')
        location_h5s = glob.glob(PREV_LOCATION_PATH + '/*.h5')
        for i in range(num_players):
            ind = 0 if i < NUM_TIMES_TO_LOAD_BEST else i - NUM_TIMES_TO_LOAD_BEST + 1
            print(f'loaded {draw_h5s[ind]} at position {i}')
            draw_agent[i].load_model(draw_h5s[ind])
            flip_agent[i].load_model(flip_h5s[ind])
            location_agent[i].load_model(location_h5s[ind])
        results = None
        with open(os.path.join(RESULTS_PATH, f"{PREV_ROUND}.json"), 'rb') as f:
            results = json.load(f)
        if 'epsilons' in results:
            # print('loaded epsilons:', results['epsilons'])
            epsilon = results['epsilons'][0]
            new_epsilons = []
            if epsilon > MAX_EPSILON_RESET:
                new_epsilons = [epsilon for _ in range(num_players)]
            else:
                offset = (MAX_EPSILON_RESET - epsilon) / (num_players - 1)
                for _ in range(num_players):
                    new_epsilons.append(epsilon)
                    epsilon += offset
            epsilons = new_epsilons
            print('loaded_epsilons:', new_epsilons)
            
            
                
            
            
            # # for i in NUM_TIMES_TO_LOAD_BEST:
            #     # new_epsilons.append(epsilon)
            # # remaining
            # for i in range(num_players):
            #     new_epsilons.append(epsilon)
        
        # time.sleep(5)
        time.sleep(2)
        
    return epsilons, draw_agent, flip_agent, location_agent


if __name__ == '__main__':
    os.makedirs(DRAW_PATH, exist_ok=True)
    os.makedirs(FLIP_PATH, exist_ok=True)
    os.makedirs(LOCATION_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    
    
    # num_players = NUM_PLAYERS
    # env = gym.make('skyjo-v0', num_agents=num_players)
    
    
    
    epsilon_decay = 0.9999
    discount_factor = 0.9
    # learning_rate = 0.1
    learning_rate = 0.00001
    replay_buffer_size = 10
    # batch_size = 1 #10
    batch_size = 10
    
    gamma = 0.6
    
    
    
    #TODO COUNT NUM OF TIMES EACH PLAYER DOES AN ACTION AND AT WHAT LOCATIONS AND PRINT
    # TODO BUILD INTERACTIVE INTERFACE
    # TODO PIT THE BEST AGAINST OTHER ITERATIONS AND PRINT OUT WIN PERCENTAGE OVER TIME (graph how many wins over time?)
    # sep epsilon for each agent
    
    
    
    # def __init__(self, input_size, output_size, hidden_sizes, discount_factor, learning_rate, replay_buffer_size, batch_size):
    
    
    
    # INPUT_OPTIONS = {'player': field_size,
    #                  'next_player': field_size, # * (num_players - 1),
    #                  'discard': 1,
    #                  'draw':1,
    #                  'player_amounts': 2,
    #                  'other_player_amounts': 2 * (num_players - 1)}
    # DRAW_COLS = ['player', 'discard', 'player_amounts', 'other_player_amounts']
    # FLIP_COLS = ['player', 'draw', 'player_amounts', 'other_player_amounts']
    # LOCATION_COLS_D = ['player', 'next_player', 'discard', 'player_amounts', 'other_player_amounts']
    # LOCATION_COLS_F = ['player', 'next_player', 'draw', 'player_amounts', 'other_player_amounts']
    
    
    
    
    # INPUT_OPTIONS = {k:v for k, v in INPUT_OPTIONS.items() if k in ['field', 'action_card', 'player_amounts', 'other_player_amounts']}
    # INPUT_SIZE = sum(INPUT_OPTIONS.values())
    # state_selection = INPUT_OPTIONS.keys()
    
    
    # draw_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in DRAW_COLS]), 2, DRAW_HIDDEN, 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)]
    # flip_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in FLIP_COLS]), 2, FLIP_HIDDEN, 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)]
    # location_agent = [DQNAgent(sum([INPUT_OPTIONS[i] for i in LOCATION_COLS_D]), field_size, LOCATION_HIDDEN, 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)]
    
    
    num_players, env, draw_agent, flip_agent, location_agent = instantiate(learning_rate, discount_factor, replay_buffer_size, batch_size)
    
    print('env inited', env.action_space, env.num_agents, env.game)

    # epsilon = 1
    epsilons = [1 for _ in range(num_players)]
    
    epsilons, draw_agent, flip_agent, location_agent = load_previous(num_players, epsilons, draw_agent, flip_agent, location_agent)
    
    # possible_moves = 
    def reset_player_stats(num_players, field_size):
        # player_stats = {i: {'wins':0, 'ended':0, 'discard': 0, 'draw': 0, 'flip': 0, 'location': {i for i in range(field_size)}, 'result_order':[]} for i in range(num_players)}
        # player_stats = {i: {'wins':0, 'ended':0, 'discard': 0, 'draw': 0, 'flip': 0, 'location': [0 for _ in range(field_size)], 'result_order':[]} for i in range(num_players)}
        player_stats = {i: {'wins':0, 'ended':0, 'ended_win':0, 'discard': 0, 'draw': 0, 'flip': 0, 'location': [0 for _ in range(field_size)], 'result_order':[], 'final_results':[], 'final_results_raw':[], 'final_results_actual':[], 'num_na':[]} for i in range(num_players)}
        
        
        return player_stats

    player_stats = reset_player_stats(num_players, field_size)
    
    total_rewards = [[] for _ in range(num_players)]
    last_save = 0
    for episode in range(MAX_EPISODES):
        state = env.reset()
        print(state)
        round_rewards = [0 for _ in range(num_players)]
        
        round_iter = 0
        for i in range(MAX_TRY):
            curr_player = state['current_player']
            if curr_player == 0:
                round_iter += 1
            print('\nPlayer:', curr_player, 'is_done:', env.is_done())
            draw_info = ActionInfo(state, DRAW_COLS) #, 'player', 'discard')
            flip_info = ActionInfo(state, FLIP_COLS) #'player', 'draw', is_run=False)          
            
            # print('state', state)
            
            action = [0, 0]
            flip_action = 0 
            
            draw_action = draw_info.calc_action(draw_agent[curr_player], epsilons[curr_player])
            
            if draw_action == 1:
                flip_action = flip_info.calc_action(flip_agent[curr_player], epsilons[curr_player])
            
            loc_cols = LOCATION_COLS_D if flip_action == 0 else LOCATION_COLS_F
            location_info = ActionInfo(state, loc_cols) #, 'player', 'discard' if flip_action == 0 else 'draw')    
            mask = state['legal_replace'] if flip_action == 0 else state['legal_flip']
            
            
            
            action[0] = draw_action + flip_action
            
            location = location_info.calc_action(location_agent[curr_player], epsilons[curr_player], np.array(mask).flatten())
            action[1] = location
            
            if draw_action == 0:
                player_stats[curr_player]['discard'] += 1
            elif flip_action == 0:
                player_stats[curr_player]['draw'] += 1
            else:
                player_stats[curr_player]['flip'] += 1
            
            player_stats[curr_player]['location'][location] += 1
            
            # next_state, reward, done, _ = env.step(action_draw, action_flip, action_location)
            next_state, reward, done, _, _ = env.step(action)
            
            if i > WARM_UP_TIME:
                reward -= (TIME_REWARD_MULT * (i - WARM_UP_TIME))
            
            if i >= MAX_TRY - num_players and not env.is_done():
                reward = NOT_FINISHING_REWARD - abs(env.game.round.current_player.get_score(include_unknowns=True))
                # reward = -100
            print('reward', reward, 'done', done)
            
            if done:
                player_stats[env.game.round.player_id_that_ended]['ended'] += 1
                winners = env.game.round.winners
                final_results = env.game.round.final_results
                final_results_raw = env.game.round.final_results_raw
                final_results_actual = env.game.round.final_results_actual
                na_count = env.game.round.na_count
                for j, (res, res_raw, res_act, na_cnt) in enumerate(zip(final_results, final_results_raw, final_results_actual, na_count)):
                    if res_raw > 142 or (na_cnt > 0 and na_cnt % 3 != 0): # this should be impossible
                        print('an issue occured:')
                        print(j, res_raw, na_cnt)
                        env.game.round.players[j].print_field(peak=True)
                        print('no peak')
                        env.game.round.players[j].print_field(peak=False)
                        
                        exit(0)
                    player_stats[j]['final_results'].append(res)
                    player_stats[j]['final_results_raw'].append(res_raw)
                    player_stats[j]['final_results_actual'].append(res_act)
                    player_stats[j]['num_na'].append(na_cnt)
                    
                    # if na_cnt > 0 and na_cnt % 3 != 0:
                        # env.game.round.players[j]
                    
                    
                # for j in range(num_players):
                    if j in winners:
                        player_stats[j]['result_order'].append(1)
                        player_stats[j]['wins'] += 1
                        if j == env.game.round.player_id_that_ended:
                            player_stats[j]['ended_win'] += 1
                    else:
                        player_stats[j]['result_order'].append(0)
                        
                    # player_stats[j]['result_order'].append(0 if j not in winners else 1)

                break
            
            # print('next_state', next_state)
            draw_info.set_outputs(next_state, reward, done)
            flip_info.set_outputs(next_state, reward, done)
            location_info.set_outputs(next_state, reward, done)
            
            draw_agent[curr_player].store_experience(draw_info.get_tuple())
            draw_agent[curr_player].train()
            if flip_info.is_run:
                flip_agent[curr_player].store_experience(flip_info.get_tuple())
                flip_agent[curr_player].train()
            location_agent[curr_player].store_experience(location_info.get_tuple())
            location_agent[curr_player].train()
            
            round_rewards[curr_player] += reward
            
            q_values = [0, 0]
            
            
            
            if round_iter % EPSILON_DECAY_FREQUENCY == 0:
                epsilons[curr_player] *= epsilon_decay
            # if i > 0 and curr_player == 0:
                # epsilons[] *= epsilon_decay
                
            state = next_state
        
            
        env.render()
        for i, val in enumerate(round_rewards):
            total_rewards[i].append(val)
            
        last_save += 1
            
        if episode % UPDATE_FREQUENCY == 0:
            print(f'\n\nepisode: {episode} / {MAX_EPISODES}')
            time.sleep(1)
        
        if episode % SAVE_FREQUENCY == 0 and episode > 0:
            save_process(total_rewards, epsilons, num_players, draw_agent, flip_agent, location_agent, player_stats)
            total_rewards = [[] for _ in range(num_players)]
            PREV_DRAW_PATH = DRAW_PATH
            PREV_FLIP_PATH = FLIP_PATH
            PREV_LOCATION_PATH = LOCATION_PATH
            PREV_ROUND = TRAIN_ROUND
            
            TRAIN_ROUND += 1
            DRAW_PATH = f'{PARENT_FOLDER}/draw_agent/{TRAIN_ROUND}'
            FLIP_PATH = f'{PARENT_FOLDER}/flip_agent/{TRAIN_ROUND}'
            LOCATION_PATH = f'{PARENT_FOLDER}/location_agent/{TRAIN_ROUND}'
            RESULTS_PATH = f'{PARENT_FOLDER}/results/'
            os.makedirs(DRAW_PATH, exist_ok=True)
            os.makedirs(FLIP_PATH, exist_ok=True)
            os.makedirs(LOCATION_PATH, exist_ok=True)
            os.makedirs(RESULTS_PATH, exist_ok=True)
            
            player_stats = reset_player_stats(num_players, field_size)
            
            epsilons, draw_agent, flip_agent, location_agent = load_previous(num_players, epsilons, draw_agent, flip_agent, location_agent)
            last_save = 0
            
# def save_process(total_rewards, epsilon, num_players, draw_agent, flip_agent, location_agent):
    if last_save > 0:
        save_process(total_rewards, epsilons, num_players, draw_agent, flip_agent, location_agent, player_stats)
        
    # for i in order:
    #     plt.plot(total_rewards[i], label=f'player {i}')
    # plt.legend()
    # # plt.show()
    # # plt.imsave(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.png'), plt.gcf())
    # plt.savefig(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.png'))
        # json.dump(res, f)
        
        
        # f.write(f'total_rewards:{total_rewards}')
        