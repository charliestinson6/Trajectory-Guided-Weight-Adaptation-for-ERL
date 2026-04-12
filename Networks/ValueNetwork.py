import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Neural Network model
class ValueNetwork(nn.Module):
    """
    The value network for Soft Actor-Critic agents
    """
    def __init__(self, device, state_space_dim, hidden_dims=(64, 64), activation_fc=F.relu, seed=None):
        super(ValueNetwork, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims)
        if seed is not None:
            torch.manual_seed(seed)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(state_space_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.device = device
        self.to(device)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        nn_output = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            nn_output = self.activation_fc(hidden_layer(nn_output))
        nn_output = self.output_layer(nn_output)
        return nn_output