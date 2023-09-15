import torch
from torch import nn
from torch import optim
import random
from collections import deque
import numpy as np


class Agent(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Agent, self).__init__()
        
        # Create the input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # Create the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Create the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Define the activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Pass the input through the input layer and activation function
        x = self.activation(self.input_layer(x))
        
        # Pass the input through the hidden layers and activation function
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        
        # Pass the input through the output layer
        x = self.output_layer(x)
        
        return x
    
class DQNAgent:
    def __init__(self, input_size, output_size, hidden_sizes, discount_factor, learning_rate, replay_buffer_size, batch_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
         
        self.online_network = Agent(input_size, output_size, hidden_sizes)
        
        self.target_network = Agent(input_size, output_size, hidden_sizes)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.losses = []
    
    def __call__(self, state, epsilon):
        return self.select_action(state, epsilon)
    
    def select_action(self, state, epsilon):
        q_values = []
        if random.random() < epsilon:
            q_values = np.random.rand(self.output_size)
            # action = random.randint(0, self.online_network.output_layer.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.online_network(torch.FloatTensor(state)).numpy()
                # action = q_values.argmax().item()
        return q_values
    
    def store_experience(self, tup): #state, action, reward, next_state, done):
        # print(tup)
        self.replay_buffer.append(tup) #(state, action, reward, next_state, done))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        # print("states: ", states)
        # print('next_states: ', next_states)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.online_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            max_q_values = self.target_network(next_states).max(1)[0]
            q_targets = rewards + (1 - dones) * self.discount_factor * max_q_values
        
        loss = nn.MSELoss()(q_values, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def save_model(self, filename):
        torch.save(self.online_network.state_dict(), filename)
    
    def load_model(self, filename):
        self.online_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
    
    def get_losses(self):
        # print(np.array(self.losses).shape)
        # return np.array(self.losses)
        return self.losses