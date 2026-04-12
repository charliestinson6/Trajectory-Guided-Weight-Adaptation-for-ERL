import numpy as np
import torch
import random
from collections import deque, namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "truncated"])

class ReplayBuffer:
    def __init__(self, device, seed, buffer_size=None, batch_size=None):
        random.seed(seed)
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def size(self):
        return self.memory.__len__()
    
    def add(self, state, action, reward, next_state, done, truncated):
        e = Experience(state, action, reward, next_state, done, truncated)
        self.memory.append(e)

    def sample(self, batch_size=None):
        if batch_size is not None:
            sampled_indices = random.sample(range(len(self.memory)), k=batch_size)
        else:
            sampled_indices = random.sample(range(len(self.memory)), k=self.batch_size)

        experiences = [self.memory[i] for i in sampled_indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        truncateds = torch.from_numpy(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, truncateds, sampled_indices)
    
    def clear(self):
        self.memory.clear()
    
    def rollout(self, trajectory_length=None):
        # Find the indices of all 'done' or 'truncated' flags
        end_indices = [i for i, exp in enumerate(self.memory) if exp.done or exp.truncated]
        if trajectory_length is None:
            if not end_indices:
                # If no 'done' or 'truncated' flags are found, return an empty list
                return []
            # Most recent episode start index
            if len(end_indices) >= 2:
                start_index = end_indices[-2] + 1
            else:
                start_index = 0
            episode = list(self.memory)[start_index:end_indices[-1]+1]

        else:
            episode = list(self.memory)[:trajectory_length]

        states, actions, rewards, next_states, dones, truncateds = zip(*episode)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        truncateds = torch.from_numpy(np.array(truncateds)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, truncateds)
    