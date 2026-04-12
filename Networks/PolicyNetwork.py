import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Neural Network model
class PolicyNetwork(nn.Module):
    """
    The policy network for Policy Gradient and Soft Actor-Critic agents
    """
    def __init__(self, device, state_space_dim, action_space_dim, hidden_dims=(64, 64), activation_fc=F.relu, seed=None):
        super(PolicyNetwork, self).__init__()
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
        self.output_layer = nn.Linear(hidden_dims[-1], action_space_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.to(device)

    def forward(self, state):
        # If state is a tuple/list (e.g. (20,7,0)) convert to numpy then tensor
        if isinstance(state, (tuple, list)):
            state = np.array(state)

        # Convert numpy arrays to torch tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        # If not yet a tensor, create one (handles ints, floats, etc.)
        if not torch.is_tensor(state):
            state = torch.tensor(state)

        # Ensure correct device and dtype
        state = state.to(self.device).float()

        nn_output = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            nn_output = self.activation_fc(hidden_layer(nn_output))
        nn_output = self.output_layer(nn_output)
        nn_output = self.softmax(nn_output)
        return nn_output