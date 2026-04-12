import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib.pyplot as plt
from collections import defaultdict
from IJCNN2026.Algorithms.BaseAgent import BaseAgent
from IJCNN2026.Networks.QNetwork import QNetwork
from IJCNN2026.Buffer.ReplayBuffer import ReplayBuffer
import random
import os

class DDQN(BaseAgent):
    def __init__(self, env, buffer_size=int(1e4), batch_size=1024, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_rate=0.99, tau=1e-3, lr=0.0005, hidden_dims=(64, 64),
                 activation_fc=F.relu, seed=None, env_seed=None):
        super().__init__(env, seed)
        self.simulation_seed = seed + 1000 if seed is not None else None
        self.env_seed = env_seed

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = epsilon_start

        # Flag to identify if agent is updated after every step or at the end of every episode
        self.update_mode = "step"
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.q_network = QNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        self.q_network_target = QNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        self.optimiser = optim.Adam(self.q_network.parameters(), lr)

        self.buffer = ReplayBuffer(device, self.seed, buffer_size=buffer_size, batch_size=batch_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.episode_rewards = []
        self.cumulative_rewards = []

    def train(self, training_episodes=None):
        # ------------------------------------------------------------
        # Determine base seed for environment resets
        # ------------------------------------------------------------
        is_dynamic = hasattr(self.env.unwrapped, 'dynamic_obstacle') and self.env.unwrapped.dynamic_obstacle

        if hasattr(self.env.unwrapped, 'seed') and self.env.unwrapped.seed is not None:
            base_seed = self.env.unwrapped.seed
        elif hasattr(self, 'env_seed') and self.env_seed is not None:
            base_seed = self.env_seed
        else:
            base_seed = getattr(self, 'seed', None)

        self.current_obstacle_seed = base_seed

        for episode in range(1, training_episodes + 1):
            # ------------------------------------------------------------
            # Episode-specific seed (independent, reproducible episodes)
            # ------------------------------------------------------------
            seed = base_seed + episode - 1 if base_seed is not None else None

            # ------------------------------------------------------------
            # Use CURRENT regime for environment interaction
            # ------------------------------------------------------------
            active_seed = self.current_obstacle_seed if is_dynamic else seed

            # Reset environment using current regime
            state, _ = self.env.reset(seed=active_seed)

            # ------------------------------------------------------------
            # AFTER using the regime, decide whether to update for next episode
            # ------------------------------------------------------------
            if is_dynamic:
                if episode % self.env.unwrapped.update_frequency == 0:
                    self.current_obstacle_seed = seed

            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                self.buffer.add(state, action, reward, next_state, done, truncated)

                episode_reward += reward

                state = next_state

                if len(self.buffer.memory) >= self.buffer.batch_size:
                    self.update()

            self.episode_rewards.append(episode_reward)
            if self.cumulative_rewards:
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + episode_reward)
            else:
                self.cumulative_rewards.append(episode_reward)
            
            self.decay_epsilon(episode)

            if (episode / training_episodes * 100).is_integer():
                if hasattr(self.env.unwrapped, 'seed'):
                    print(f'Alg Seed = {self.seed}, Env Seed = {self.env.unwrapped.seed}: {int(episode / training_episodes * 100)}% Complete')
                else:
                    print(f'{int(episode / training_episodes * 100)}% Complete')

    def get_action(self, state, deterministic=False):
        """Select an action using epsilon-greedy policy based on the current Q-network."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()

        if not deterministic and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return action_values.argmax(dim=1).item()
            
    def return_action_probs(self, state):
        if isinstance(state, int) or isinstance(state, float):
            state = np.array([state])
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        action_values = action_values.cpu().numpy().flatten()
        n_actions = self.env.action_space.n

        # Greedy action index
        best_action = np.argmax(action_values)

        # Epsilon-greedy probabilities
        action_probs = np.ones(n_actions) * (self.epsilon / n_actions)
        action_probs[best_action] += (1.0 - self.epsilon)

        return action_probs

    def update(self, batch=None):
        """Update the Q-network based on sampled experiences from the replay buffer."""
        if batch is None:
            batch = self.buffer.sample()
        states, actions, rewards, next_states, dones, truncateds, _ = batch
        states = torch.from_numpy(np.vstack(states.cpu())).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions.cpu())).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards.cpu())).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states.cpu())).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones.cpu()).astype(np.uint8)).float().to(self.device)

        next_actions = self.q_network(states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, next_actions).detach()
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_current = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(Q_current, Q_targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.soft_update(self.q_network, self.q_network_target)

    def soft_update(self, model, target_model):
        """Soft update model parameters."""
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self, current_episode):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay_rate ** current_episode))

    def track_episode_rewards(self, show_plot=True):
        """Track episode rewards and optionally plot them."""
        if show_plot:
            x = list(range(1, len(self.episode_rewards) + 1))
            plt.plot(x, self.episode_rewards, color='darkblue', alpha=0.3)
            plt.plot(x, self.episode_rewards, marker='o', markersize=8, markerfacecolor='white', markeredgecolor='blue', linestyle='')
            plt.xlabel('Episode')
            plt.title('Episode Reward')
            plt.show()

        return self.episode_rewards

    def track_cumulative_rewards(self, show_plot=True):
        """Track cumulative rewards and optionally plot them."""
        if show_plot:
            x = list(range(1, len(self.cumulative_rewards) + 1))
            plt.plot(x, self.cumulative_rewards)
            plt.xlabel('Episode')
            plt.title('Cumulative Reward')
            plt.show()

        return self.cumulative_rewards

    def simulate_returns(self, seed):
        """Simulate episode using the current policy to estimate returns for ensemble weighting."""
        returns = []

        state, _ = self.env.reset(seed=seed)

        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            action = self.get_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            episode_reward += reward
            state = next_state

        returns.append(episode_reward)

        return np.array(returns)