import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.distributions as distributions
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from collections import deque, namedtuple
from copy import deepcopy
from IJCNN2026.Algorithms.BaseAgent import BaseAgent
from IJCNN2026.Networks.QNetwork import QNetwork
from IJCNN2026.Networks.PolicyNetwork import PolicyNetwork
from IJCNN2026.Buffer.ReplayBuffer import ReplayBuffer


class SAC(BaseAgent):
    def __init__(self, env, buffer_size=int(1e4), batch_size=1024, gamma=0.99, tau=1e-3, actor_lr=0.0005, critic1_lr=0.0005,
                 critic2_lr=0.0005, log_alpha_lr=0.0005, actor_hidden_dims=(64, 64), critic_hidden_dims=(64, 64), min_alpha=None, 
                 activation_fc=F.relu, seed=None, env_seed=None):
        super().__init__(env, seed)
        self.simulation_seed = seed + 1000 if seed is not None else None
        self.env_seed = env_seed

        self.gamma = gamma
        self.tau = tau
        self.clip_grad_param = 1.0
        self.lr = actor_lr
        self.log_alpha_lr = log_alpha_lr

        # Flag to identify if agent is updated after every step or at the end of every episode
        self.update_mode = "step"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = PolicyNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=actor_hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), actor_lr)

        self.critic1 = QNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=critic_hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        self.critic2 = QNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=critic_hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)

        self.critic1_optimiser = optim.Adam(self.critic1.parameters(), critic1_lr)
        self.critic2_optimiser = optim.Adam(self.critic2.parameters(), critic2_lr)

        self.target_entropy = -env.action_space.n

        # Flag to indicate if alpha is fixed or learned
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach().to(self.device)
        self.alpha_optimiser = optim.Adam(params=[self.log_alpha], lr=log_alpha_lr)
        self.min_alpha = min_alpha

        self.buffer = ReplayBuffer(device, seed, buffer_size=buffer_size, batch_size=batch_size)
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

                if self.buffer.size() >= self.buffer.batch_size:
                    self.update()
            self.episode_rewards.append(episode_reward)
            if self.cumulative_rewards:
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + episode_reward)
            else:
                self.cumulative_rewards.append(episode_reward)

            if (episode / training_episodes * 100).is_integer():
                if hasattr(self.env.unwrapped, 'seed'):
                    print(f'Alg Seed = {self.seed}, Env Seed = {self.env.unwrapped.seed}: {int(episode / training_episodes * 100)}% Complete')
                else:
                    print(f'{int(episode / training_episodes * 100)}% Complete')
                    
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            action_probs = self.actor.forward(state)
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                dist = distributions.Categorical(action_probs)
                action = dist.sample().item()

        return action

    def update(self, batch=None):
        if batch is None:
            batch = self.buffer.sample()
        states, actions, rewards, next_states, dones, truncateds, _ = batch

        self.update_critics(states, actions, rewards, next_states, dones)
        self.update_actor(states)

    def update_actor(self, states):
        action_probs = self.actor.forward(states)
        log_pis = torch.log(action_probs + 1e-8)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)

        actor_loss = (action_probs * (self.alpha * log_pis - min_Q)).sum(1).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.alpha = self.log_alpha.exp().detach().to(self.device)
        if self.min_alpha is not None:
            self.alpha = torch.max(self.alpha, torch.tensor(self.min_alpha).to(self.device))

    def update_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor.forward(next_states)
            log_pis = torch.log(action_probs + 1e-8)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())

        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)

        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        self.critic1_optimiser.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimiser.step()

        self.critic2_optimiser.zero_grad()
        critic2_loss.backward()
        self.critic2_optimiser.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def return_action_probs(self, state):
        if isinstance(state, int) or isinstance(state, float):
            state = np.array([state])

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = np.array(self.actor.forward(state_tensor)[0].cpu()) if self.device.type == "cuda" else np.array(self.actor.forward(state_tensor)[0])

        return action_probs

    def track_episode_rewards(self, show_plot=True):
        if show_plot:
            x = list(range(1, len(self.episode_rewards) + 1))
            plt.plot(x, self.episode_rewards, color='darkblue', alpha=0.3)
            plt.plot(x, self.episode_rewards, marker='o', markersize=8, markerfacecolor='white', markeredgecolor='blue', linestyle='')
            plt.xlabel('Episode')
            plt.title('Episode Reward')
            plt.show()

        return self.episode_rewards

    def track_cumulative_rewards(self, show_plot=True):
        if show_plot:
            x = list(range(1, len(self.cumulative_rewards) + 1))
            plt.plot(x, self.cumulative_rewards)
            plt.xlabel('Episode')
            plt.title('Cumulative Reward')
            plt.show()

        return self.cumulative_rewards
    
    def simulate_returns(self, seed):
        """Simulate episodes and store experiences in temp_buffer."""
        returns = []

        self.simulation_seed += 1 if self.simulation_seed is not None else None
        #state, _ = self.env.reset(seed= self.simulation_seed)
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
    