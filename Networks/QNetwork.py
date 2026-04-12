import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Define the Neural Network model
class QNetwork(nn.Module):
    """
    The neural network for DQN agents
    """
    def __init__(self, device, state_space_dim, action_space_dim, hidden_dims=(64, 64), activation_fc=F.relu, seed=None):
        super(QNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(state_space_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], action_space_dim)

        self.device = device
        self.to(device)

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(self.hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        nn_output = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            nn_output = self.activation_fc(hidden_layer(nn_output))
        nn_output = self.output_layer(nn_output)
        return nn_output