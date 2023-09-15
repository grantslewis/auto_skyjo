import gym
# import gym_game
import skyjo_gym_game
import numpy as np
import random

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
    


if __name__ == '__main__':
    env = gym.make('skyjo-v0', num_agents=4)
    print(env.action_space, env.num_agents, env.game)
    MAX_EPISODES = 1000
    MAX_TRY = 500
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    
    num_players = 2
    
    # q_table_draw = [np.zeros(((field_size + 1) * num_vals , 2)) for _ in range(num_players)]
    # q_table_draw = [np.zeros(((field_size + 1 ) * num_vals , 2)) for _ in range(num_players)]
    q_table_draw = [np.zeros((*[num_vals for _ in range(field_size)] , 2)) for _ in range(num_players)]
    
    print(q_table_draw[0].shape)
    
    
    
    q_table_flip = [np.zeros(((field_size + 1) * num_vals , 2)) for _ in range(num_players)]
    
    q_table_location = [np.zeros(((field_size + 1) * num_vals , field_size)) for _ in range(num_players)]
    
    
    # replace_action = np.zeros(((field_size * 2) + 1) * num_vals)
    replace_action = np.zeros(((field_size + 1) * num_vals, field_size) )
    
    
    
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        print(state)
        total_rewards = [0 * num_players]
        
        
        for i in range(MAX_TRY):
            print(q_table_draw[0].shape)
            print(q_table_draw[0])
            print(q_table_draw[0][0,1,2,3,4,5,6,7,8,9,10,11,12])
            
            
            one_hot = state_to_one_hot(state['player'])
            one_hot = np.stack((one_hot, state_to_one_hot(state['discard'])), axis=0) # 1
            print('one_hot:' , one_hot)
            
            
            
            # # one_hot_state = state_to_one_hot(state)
            # # print(one_hot_state)
            # curr_player = env.current_player()
            # action_draw = proceed_action(one_hot_state, q_table_draw[curr_player], epsilon)
            # action_flip = 0
            # ret_actions = [action_draw, 0]
            
            # if action_draw == 1:
            #     one_hot_state[-1, state[2]] = 0
            #     one_hot_state[-1, state[3]] = 1
            
            #     action_flip = proceed_action(one_hot_state, q_table_flip[curr_player], epsilon)
                
                
            
            # action_location = location_action(one_hot_state, q_table_location[curr_player], epsilon)
            
            # next_state, reward, done, _ = env.step(action_draw, action_flip, action_location)
            # total_rewards[curr_player] += reward
            
            # next_one_hot_state = state_to_one_hot(next_state)
            # q_table_draw[curr_player] = update_q_table(one_hot_state, action_draw, reward, next_one_hot_state, q_table_draw[curr_player], learning_rate, gamma)
            # if action_draw == 1:
            #     q_table_flip[curr_player] = update_q_table(one_hot_state, action_flip, reward, next_one_hot_state, q_table_flip[curr_player], learning_rate, gamma)
            # q_table_location[curr_player] = update_q_table(one_hot_state, action_location, reward, next_one_hot_state, q_table_location[curr_player], learning_rate, gamma)
            
            # state = next_state
            
            
            # env.step(env.action_space.sample()) # take a random action
            
            env.render()
            
            