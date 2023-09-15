import gym
# import gym_game
import skyjo_gym_game
import numpy as np
import random
from agent import Agent, DQNAgent

field_size = 12
num_vals = 17

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
    num_players = 4
    env = gym.make('skyjo-v0', num_agents=num_players)
    print(env.action_space, env.num_agents, env.game)
    MAX_EPISODES = 10 #1000
    MAX_TRY = 500
    epsilon = 1
    epsilon_decay = 0.999
    discount_factor = 0.9
    learning_rate = 0.1
    replay_buffer_size = 100
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
    
    
    
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        print(state)
        total_rewards = [0 for _ in range(num_players)]
        
        
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
            
            total_rewards[curr_player] += reward
            
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
            # # total_rewards[curr_player] += reward
            
            # # next_one_hot_state = state_to_one_hot(next_state)
            # # q_table_draw[curr_player] = update_q_table(one_hot_state, action_draw, reward, next_one_hot_state, q_table_draw[curr_player], learning_rate, gamma)
            # # if action_draw == 1:
            # #     q_table_flip[curr_player] = update_q_table(one_hot_state, action_flip, reward, next_one_hot_state, q_table_flip[curr_player], learning_rate, gamma)
            # # q_table_location[curr_player] = update_q_table(one_hot_state, action_location, reward, next_one_hot_state, q_table_location[curr_player], learning_rate, gamma)
            
            # # state = next_state
            
            
            # # env.step(env.action_space.sample()) # take a random action
            
            env.render()
            
        
            
            