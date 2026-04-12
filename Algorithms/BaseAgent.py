from abc import ABC, abstractmethod
from gymnasium.spaces import Discrete, Box, MultiBinary
import random
import numpy as np

# Define the BaseAgent abstract class
class BaseAgent(ABC):
    def __init__(self, env, seed=None):
        self.env = env
        self.env.action_space.seed(seed)
        
        if isinstance(env.observation_space, Discrete):
            self.state_space_dim = 1
            self.state_space_type = 'Discrete'
        elif isinstance(env.observation_space, MultiBinary):
            self.state_space_dim = env.observation_space.shape[0]
            self.state_space_type = 'MultiBinary'
        elif isinstance(env.observation_space, Box):
            self.state_space_dim = env.observation_space.shape[0]
            self.state_space_type = 'Continuous'
        else:
            self.state_space_dim = len(env.observation_space)
            self.state_space_type = 'Tuple'  

        if isinstance(env.action_space, Box):
            self.action_space_dim = int(np.prod(env.action_space.shape))
        else:
            self.action_space_dim = env.action_space.n

        self.seed = seed

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update(self, experiences):
        pass

    @abstractmethod
    def train(self, training_episodes):
        pass