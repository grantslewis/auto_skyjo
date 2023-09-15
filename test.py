import os
import gym
import skyjo_gym_game
import numpy as np
import random
from agent import Agent, DQNAgent
import torch
import json
import glob
from game_instantiator import instantiate, ActionInfo, DRAW_COLS, FLIP_COLS, LOCATION_COLS_D, LOCATION_COLS_F

# conda activate c:/Users/grant/anaconda3/envs/.venv_byu


# PARENT_DIR = './agent_results_3'
PARENT_DIR = './agent_results_2players_old'
# PARENT_DIR = './agent_results_2'

DEST_DIR = './agent_test_results_2player_old'
TEST_NUMBER = 0

# sources = [23,23,23,23,23,23]
# sources = [23,20,15,10,5,0]
# soruces = [15, 12, 9, 6, 3, 0]
# sources = [9,9,9,9,9,9]

# sources = [55, 40, 30, 20, 10, 0]
# sources = [55, 55, 55, 55, 55, 55]
sources = [86, 20]



epsilons = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# epsilons = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
# epsilons = [0,0,0,0,0,0]
# epsilons = [0, 0.1, 0.1, 0.1, 0.1, 0.1]
# epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# epsilons = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
# epsilons = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

# epsilons = [0, 1, 1, 1, 1, 1]
# epsilons = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]



field_size = 12
num_vals = 17

    
    
epsilon_decay = 0.9999
discount_factor = 0.9
# learning_rate = 0.1
learning_rate = 0.00001
replay_buffer_size = 10
# batch_size = 1 #10
batch_size = 10


TEST_SIZE = 20
max_rounds = 500



def reset_player_stats(num_players, field_size):
    # player_stats = {i: {'wins':0, 'ended':0, 'discard': 0, 'draw': 0, 'flip': 0, 'location': {i for i in range(field_size)}, 'result_order':[]} for i in range(num_players)}
    player_stats = {i: {'wins':0, 'forced_end_wins':0, 'ended':0, 'discard': 0, 'draw': 0, 'flip': 0, 'location': [0 for _ in range(field_size)], 'result_order':[]} for i in range(num_players)}
    
    return player_stats

def calc_action(state, agent, epsilon, mask=None):
    res = agent(state, epsilon)
    # print(res)
    # res = res.detach().numpy()
    if mask is not None:
        res = res * np.array(mask).flatten()
    return np.argmax(res)




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



if __name__ == '__main__':
    while os.path.exists(os.path.join(DEST_DIR, str(TEST_NUMBER))) and len(os.listdir(os.path.join(DEST_DIR, str(TEST_NUMBER)))) > 0:
        TEST_NUMBER += 1
    
    output_folder = os.path.join(DEST_DIR, str(TEST_NUMBER))
    
    os.makedirs(output_folder, exist_ok=True)
    
    
    num_players = 6
    if len(epsilons) != num_players:
        epsilons = [0 for _ in range(num_players)]
    
    # env = gym.make('skyjo-v0', num_agents=num_players)
    
    # draw_agent = [DQNAgent(field_size + 1, 2, (24, 32, 24), 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)]
    # flip_agent = [DQNAgent(field_size + 1, 2, (24, 32, 24), 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)]
    # location_agent = [DQNAgent(field_size + 1, field_size, (128, 256, 256, 128), 
    #                        discount_factor, learning_rate, 
    #                        replay_buffer_size, batch_size) for _ in range(num_players)] 
    
    num_players, env, draw_agent, flip_agent, location_agent = instantiate(learning_rate, discount_factor, replay_buffer_size, batch_size)
    

    player_stats = reset_player_stats(num_players, field_size)
    for i in range(num_players):
        player_stats[i]['start_epsilon'] = epsilons[i]
    
    
    
    for i in range(num_players):
        draw_agent[i].load_model(glob.glob(os.path.join(PARENT_DIR, 'draw_agent', str(i), '0*.h5'))[0])#.eval()
        draw_agent[i].online_network.eval()
        draw_agent[i].target_network.eval()
        
        flip_agent[i].load_model(glob.glob(os.path.join(PARENT_DIR, 'flip_agent', str(i), '0*.h5'))[0])#.eval()
        flip_agent[i].online_network.eval()
        flip_agent[i].target_network.eval()
        
        location_agent[i].load_model(glob.glob(os.path.join(PARENT_DIR, 'location_agent', str(i), '0*.h5'))[0])#.eval()
        location_agent[i].online_network.eval()
        location_agent[i].target_network.eval()
    
        
    with torch.no_grad():
        for i in range(TEST_SIZE):
            state = env.reset()
            done = False
            round_rewards = [0 for _ in range(num_players)]
            for j in range(max_rounds):
                curr_player = state['current_player']
                
                
                draw_info = ActionInfo(state, DRAW_COLS) #, 'player', 'discard')
                flip_info = ActionInfo(state, FLIP_COLS, is_run=False) #, 'player', 'draw', is_run=False)          
                
                action = [0, 0]
                flip_action = 0 
                
                # draw_action = draw_info.calc_action(draw_agent[curr_player], epsilons[curr_player])
                draw_action = draw_info.calc_action(draw_agent[curr_player], epsilons[curr_player])
            
                if draw_action == 1:
                    flip_action = flip_info.calc_action(flip_agent[curr_player], epsilons[curr_player])
                
                # if draw_action == 1:
                #     flip_action = flip_info.calc_action(flip_agent[curr_player], epsilons[curr_player])
                
                # location_info = ActionInfo(state, 'player', 'discard' if flip_action == 0 else 'draw')    
                loc_cols = LOCATION_COLS_D if flip_action == 0 else LOCATION_COLS_F
                location_info = ActionInfo(state, loc_cols)
                mask = state['legal_replace'] if flip_action == 0 else state['legal_flip']
                
                
                
                action[0] = draw_action + flip_action
                
                location = location_info.calc_action(location_agent[curr_player], epsilons[curr_player], mask.flatten())
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
                
                if done:
                    player_stats[env.game.round.player_id_that_ended]['ended'] += 1
                    winners = env.game.round.winners
                    for j in range(num_players):
                        if j in winners:
                            player_stats[j]['result_order'].append(1)
                            player_stats[j]['wins'] += 1
                        else:
                            player_stats[j]['result_order'].append(0)
                            
                        # player_stats[j]['result_order'].append(0 if j not in winners else 1)
    
                    break
                
                state = next_state
                
            if not done:
                winners = []
                results = [p.get_score(include_unknowns=True) for p in env.game.round.players]
                min_score = min(results)
                for r in results:
                    if r == min_score:
                        winners.append(results.index(r))
                
                for j in range(num_players):
                    player_stats[j]['result_order'].append(-1)
                    if j in winners:
                        player_stats[j]['forced_end_wins'] += 1
                    
    
                    
                
                # actions = []
                #     draw_action = draw_agent[j](state[j], epsilons[j])
                #     flip_action = flip_agent[j](state[j], epsilons[j])
                #     location_action = location_agent[j](state[j], epsilons[j])
                #     actions.append((draw_action, flip_action, location_action))
                # state, reward, done, info = env.step(actions)
                # env.render()
                # epsilons = [epsilon * epsilon_decay for epsilon in epsilons]
                
                
                
            env.close()
    for i in range(num_players):
        player_stats[i]['end_epsilon'] = epsilons[i]
        print(i, '- wins:', player_stats[i]['wins'])
        print(i, '- forced_end_wins:', player_stats[i]['forced_end_wins'])
        
        
        with open(os.path.join(output_folder, str(i) + '.json'), 'w+') as f:
            f.write(json.dumps(player_stats[i]))
            # json.dump(player_stats[i], f)
    
    
    
    
    # for i in range(100):
    