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

field_size = 12
num_vals = 17

TRAIN_ROUND = 13
OVERRIDE_PREV = False #True

if not OVERRIDE_PREV:
    while os.path.exists(f'./agent_results/draw_agent/{TRAIN_ROUND}') and len(os.listdir(f'./agent_results/draw_agent/{TRAIN_ROUND}')) > 0:
        TRAIN_ROUND += 1

DRAW_PATH = f'./agent_results/draw_agent/{TRAIN_ROUND}'
FLIP_PATH = f'./agent_results/flip_agent/{TRAIN_ROUND}'
LOCATION_PATH = f'./agent_results/location_agent/{TRAIN_ROUND}'
RESULTS_PATH = f'./agent_results/results/'

PREV_ROUND = None
PREV_ROUND = TRAIN_ROUND - 1 if PREV_ROUND is None else PREV_ROUND
PREV_DRAW_PATH = f'./agent_results/draw_agent/{PREV_ROUND}'
PREV_FLIP_PATH = f'./agent_results/flip_agent/{PREV_ROUND}'
PREV_LOCATION_PATH = f'./agent_results/location_agent/{PREV_ROUND}'
NUM_TIMES_TO_LOAD_BEST = 2

UPDATE_FREQUENCY = 10





def proceed_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.sample_draw()
    else:
        action = np.argmax(q_table[state])
    return action

def location_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action
        
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

def calc_action(state, agent, epsilon, mask=None):
    res = agent(state, epsilon)
    # print(res)
    # res = res.detach().numpy()
    if mask is not None:
        res = res * np.array(mask).flatten()
    return np.argmax(res)




class ActionInfo:
    def __init__(self, state, main_col, append_col, is_run=True) -> None:
        self.main_col = main_col
        self.append_col = append_col
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
        sel_state = state[self.main_col].flatten().tolist()
        append_col = self.append_col if append_col is None else append_col
        sel_state.append(state[append_col])
        if is_next:
            self.next_state = sel_state
        else:
            self.state = sel_state
    
    def get_tuple(self):
        if not self.is_run:
            return None
        return (self.state, self.action, self.reward, self.next_state, self.done)
    
    def calc_action(self, agent, epsilon, mask=None):
        self.is_run = True
        self.action = calc_action(self.state, agent, epsilon, mask)
        return self.action

    
    


if __name__ == '__main__':
    os.makedirs(DRAW_PATH, exist_ok=True)
    os.makedirs(FLIP_PATH, exist_ok=True)
    os.makedirs(LOCATION_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    
    
    num_players = 6
    env = gym.make('skyjo-v0', num_agents=num_players)
    print(env.action_space, env.num_agents, env.game)
    MAX_EPISODES = 50 #1500 # 50 #1000
    MAX_TRY = 500
    epsilon = 1
    epsilon_decay = 0.9999
    discount_factor = 0.9
    learning_rate = 0.1
    replay_buffer_size = 10
    # batch_size = 1 #10
    batch_size = 10
    
    gamma = 0.6
    
    
    
    
    
    # def __init__(self, input_size, output_size, hidden_sizes, discount_factor, learning_rate, replay_buffer_size, batch_size):
    
    
    draw_agent = [DQNAgent(field_size + 1, 2, (20,20), 
                           discount_factor, learning_rate, 
                           replay_buffer_size, batch_size) for _ in range(num_players)]
    flip_agent = [DQNAgent(field_size + 1, 2, (20,20), 
                           discount_factor, learning_rate, 
                           replay_buffer_size, batch_size) for _ in range(num_players)]
    location_agent = [DQNAgent(field_size + 1, field_size, (128, 256, 128), 
                           discount_factor, learning_rate, 
                           replay_buffer_size, batch_size) for _ in range(num_players)]
    
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
        if 'epsilon' in results:
            print('loaded epsilon:', results['epsilon'])
            epsilon = results['epsilon']
        
        time.sleep(5)
        
        
    
    
    # draw_agent = [Agent(field_size + 1, num_vals, (5), epsilon, epsilon_decay, learning_rate, gamma) for _ in range(num_players)]
    # flip_agent = [Agent(field_size + 1, num_vals, (5), epsilon, epsilon_decay, learning_rate, gamma) for _ in range(num_players)]
    # location_agent = [Agent(field_size + 1, num_vals, (10), epsilon, epsilon_decay, learning_rate, gamma) for _ in range(num_players)]
    
    
    # # q_table_draw = [np.zeros(((field_size + 1) * num_vals , 2)) for _ in range(num_players)]
    # # q_table_draw = [np.zeros(((field_size + 1 ) * num_vals , 2)) for _ in range(num_players)]
    # q_table_draw = [np.zeros((*[num_vals for _ in range(field_size)] , 2)) for _ in range(num_players)]
    
    # print(q_table_draw[0].shape)
    
    
    
    # q_table_flip = [np.zeros(((field_size + 1) * num_vals , 2)) for _ in range(num_players)]
    
    # q_table_location = [np.zeros(((field_size + 1) * num_vals , field_size)) for _ in range(num_players)]
    
    
    # # replace_action = np.zeros(((field_size * 2) + 1) * num_vals)
    # replace_action = np.zeros(((field_size + 1) * num_vals, field_size) )
    
    
    
    
    total_rewards = [[] for _ in range(num_players)]
    for episode in range(MAX_EPISODES):
        if episode % UPDATE_FREQUENCY == 0:
            print(f'\n\nepisode: {episode} / {MAX_EPISODES}')
            time.sleep(1)
        state = env.reset()
        print(state)
        round_rewards = [0 for _ in range(num_players)]
        
        
        for i in range(MAX_TRY):
            curr_player = state['current_player']
            print('\nPlayer:', curr_player, 'is_done:', env.is_done())
            draw_info = ActionInfo(state, 'player', 'discard')
            flip_info = ActionInfo(state, 'player', 'draw', is_run=False)          
            
            action = [0, 0]
            flip_action = 0 
            
            draw_action = draw_info.calc_action(draw_agent[curr_player], epsilon)
            
            if draw_action == 1:
                flip_action = flip_info.calc_action(flip_agent[curr_player], epsilon)
            
            location_info = ActionInfo(state, 'player', 'discard' if flip_action == 0 else 'draw')    
            mask = state['legal_replace'] if flip_action == 0 else state['legal_flip']
            
            
            
            action[0] = draw_action + flip_action
            
            location = location_info.calc_action(location_agent[curr_player], epsilon, mask.flatten())
            action[1] = location
            
            # next_state, reward, done, _ = env.step(action_draw, action_flip, action_location)
            next_state, reward, done, _, _ = env.step(action)
            print('reward', reward, 'done', done)
            if done:
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
            
            
            
            
            
            if i > 0 and curr_player == 0:
                epsilon *= epsilon_decay
                
            state = next_state
        
            
            
            
            
            
            
            
            
            
            
            # print(q_table_draw[0].shape)
            # print(q_table_draw[0])
            # print(q_table_draw[0][0,1,2,3,4,5,6,7,8,9,10,11,12])
            
            
            # one_hot = state_to_one_hot(state['player'])
            # one_hot = np.stack((one_hot, state_to_one_hot(state['discard'])), axis=0) # 1
            # print('one_hot:' , one_hot)
            
            
            
            # # # one_hot_state = state_to_one_hot(state)
            # # # print(one_hot_state)
            # # curr_player = env.current_player()
            # # action_draw = proceed_action(one_hot_state, q_table_draw[curr_player], epsilon)
            # # action_flip = 0
            # # ret_actions = [action_draw, 0]
            
            # # if action_draw == 1:
            # #     one_hot_state[-1, state[2]] = 0
            # #     one_hot_state[-1, state[3]] = 1
            
            # #     action_flip = proceed_action(one_hot_state, q_table_flip[curr_player], epsilon)
                
                
            
            # # action_location = location_action(one_hot_state, q_table_location[curr_player], epsilon)
            
            # # next_state, reward, done, _ = env.step(action_draw, action_flip, action_location)
            # # round_rewards[curr_player] += reward
            
            # # next_one_hot_state = state_to_one_hot(next_state)
            # # q_table_draw[curr_player] = update_q_table(one_hot_state, action_draw, reward, next_one_hot_state, q_table_draw[curr_player], learning_rate, gamma)
            # # if action_draw == 1:
            # #     q_table_flip[curr_player] = update_q_table(one_hot_state, action_flip, reward, next_one_hot_state, q_table_flip[curr_player], learning_rate, gamma)
            # # q_table_location[curr_player] = update_q_table(one_hot_state, action_location, reward, next_one_hot_state, q_table_location[curr_player], learning_rate, gamma)
            
            # # state = next_state
            
            
            # # env.step(env.action_space.sample()) # take a random action
            
        env.render()
        for i, val in enumerate(round_rewards):
            total_rewards[i].append(val)
            
            
    summed_rewards = np.sum(total_rewards, axis=1)
    order = np.argsort(summed_rewards)[::-1].tolist()
    # winner_agent = np.argmax(np.sum(total_rewards, axis=1))
    # winner_agent = np.argmin(np.sum(total_rewards, axis=1))
    
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
    print('epsilon:', epsilon)
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
        
        
    res = {'order': order, 'summed_rewards': summed_rewards.tolist(), 'epsilon': epsilon,
           'avg_draw_losses': avg_draw_losses, 'avg_flip_losses': avg_flip_losses, 'avg_location_losses': avg_location_losses,
           'total_rewards': total_rewards,
           'draw_losses': draw_losses, 'flip_losses': flip_losses, 'location_losses': location_losses,
           }
    
    with open(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.json'), 'w') as f:
        f.write(json.dumps(res))
        
    for i in order:
        plt.plot(total_rewards[i], label=f'player {i}')
    plt.legend()
    # plt.show()
    # plt.imsave(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.png'), plt.gcf())
    plt.savefig(os.path.join(RESULTS_PATH, f'{TRAIN_ROUND}.png'))
        # json.dump(res, f)
        
        
        # f.write(f'total_rewards:{total_rewards}')
        