import torch
from torch import nn

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