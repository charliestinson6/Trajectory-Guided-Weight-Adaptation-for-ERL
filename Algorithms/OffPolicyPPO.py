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
from IJCNN2026.Networks.ValueNetwork import ValueNetwork
from IJCNN2026.Networks.PolicyNetwork import PolicyNetwork
from IJCNN2026.Buffer.ReplayBuffer import ReplayBuffer

class OffPolicyPPO(BaseAgent):
    def __init__(self, env, gamma=0.99, actor_lr=0.0005, critic_lr=0.0005, actor_hidden_dims=(64, 64), critic_hidden_dims=(64, 64), 
                 epsilon=0.1, epochs=10, lam=1, rollout_length=None, batch_size=1000, activation_fc=F.relu, seed=None, env_seed=None):
        super().__init__(env, seed)
        self.simulation_seed = seed + 1000 if seed is not None else None
        self.env_seed = env_seed

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.lam = lam
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.lr = actor_lr

        # Flag to identify if agent is updated after every step or at the end of every episode
        self.update_mode = "episode"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.buffer = ReplayBuffer(device, seed)

        # Initialise actor and critic networks
        self.actor = PolicyNetwork(
            device=device,
            state_space_dim=self.state_space_dim,
            action_space_dim=env.action_space.n,
            hidden_dims=actor_hidden_dims,
            activation_fc=activation_fc,
            seed=seed
        ).to(device)

        self.old_actor = deepcopy(self.actor)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), actor_lr)

        self.critic = ValueNetwork(
            device=self.device, state_space_dim=self.state_space_dim, 
            hidden_dims=critic_hidden_dims, activation_fc=activation_fc, seed=seed
        ).to(self.device)

        self.critic_optimiser = optim.Adam(self.critic.parameters(), critic_lr)

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

        self.online_probs = []
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

            done = False
            truncated = False
            episode_reward = 0

            while not done and not truncated:
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Given a state and action
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)[0]
                    action_prob = action_probs[action].item()
                self.online_probs.append(action_prob)

                self.buffer.add(state, action, reward, next_state, done, truncated)

                if self.rollout_length is not None:
                    if self.buffer.size() >= self.rollout_length:
                        self.update(behaviour_action_probs=self.online_probs)
                        # Clear buffer
                        self.buffer.clear()
                        self.online_probs = []

                episode_reward += reward
                state = next_state

            self.episode_rewards.append(episode_reward)
            if self.cumulative_rewards:
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + episode_reward)
            else:
                self.cumulative_rewards.append(episode_reward)

            if self.rollout_length is None:
                self.update(behaviour_action_probs=self.online_probs)
                # Clear buffer
                self.buffer.clear()
                self.online_probs = []

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
    
    def update(self, trajectory=None, behaviour_action_probs=None):
        if trajectory is None:
            trajectory = self.buffer.rollout(self.rollout_length)
        states, actions, rewards, next_states, dones, truncateds = trajectory

        states = torch.from_numpy(np.vstack(states.cpu())).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions.cpu())).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards.cpu())).float().to(self.device).reshape(-1)
        next_states = torch.from_numpy(np.vstack(next_states.cpu())).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones.cpu()).astype(np.uint8)).float().to(self.device).reshape(-1)
        truncateds = torch.from_numpy(np.vstack(truncateds.cpu()).astype(np.uint8)).float().to(self.device).reshape(-1)
        behaviour_action_probs = torch.Tensor(behaviour_action_probs).float().to(self.device).unsqueeze(1).detach()
        behaviour_action_probs = behaviour_action_probs[:len(actions)]

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)

            # Bootstrap values for delta
            next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])
            if dones[-1] == 1.0:
                next_values[-1] = 0.0
            else:
                next_values[-1] = self.critic(next_states[-1].unsqueeze(0)).squeeze()

            # Importance sampling ratios
            old_action_probs = self.old_actor(states).gather(1, actions)
            log_old_action_probs = torch.log(old_action_probs + 1e-8)
            log_behaviour_action_probs = torch.log(behaviour_action_probs + 1e-8)
            rho = (log_old_action_probs - log_behaviour_action_probs).exp().squeeze()
            rho_clipped = torch.clamp(rho, max=1.0)
            c = torch.clamp(rho, max=1.0)

            # V-trace targets
            delta = rho_clipped * (rewards + self.gamma * next_values * (1 - dones) - values)
            v_trace_targets = values.clone()
            for t in reversed(range(len(rewards) - 1)):
                v_trace_targets[t] = values[t] + delta[t] + self.gamma * self.lam * c[t] * (v_trace_targets[t + 1] - values[t + 1])
            v_trace_targets[-1] = values[-1] + delta[-1]

            # Final bootstrap for advantages
            if dones[-1].item() == 1.0:
                final_bootstrap = torch.tensor([0.0], device=self.device)
            else:
                final_bootstrap = self.critic(next_states[-1].unsqueeze(0)).squeeze().unsqueeze(0)

            next_v_trace_targets = torch.cat([v_trace_targets[1:], final_bootstrap])
            advantages = (rewards + self.gamma * next_v_trace_targets * (1 - dones) - values).reshape(-1)

        # Perform updates using samples from the trajectory
        trajectory_size = len(rewards)

        for _ in range(self.epochs):
            # Shuffle all indices
            indices = np.arange(trajectory_size)
            np.random.shuffle(indices)

            # Break into mini-batches
            for start in range(0, trajectory_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Gather sampled data
                batch_states = states[batch_indices]
                batch_behaviour_action_probs = behaviour_action_probs[batch_indices]
                batch_action_probs = self.actor.forward(batch_states).gather(1, actions[batch_indices])
                batch_old_action_probs = self.old_actor.forward(batch_states).gather(1, actions[batch_indices])
                batch_v_trace_targets = v_trace_targets[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Ratios
                prob_ratios = (batch_action_probs / (batch_behaviour_action_probs + 1e-8)).squeeze()
                old_prob_ratios = (batch_old_action_probs / (batch_behaviour_action_probs + 1e-8)).squeeze()
                clips_upper = old_prob_ratios * (1 + self.epsilon)
                clips_lower = old_prob_ratios * (1 - self.epsilon)
                clipped_prob_ratios = torch.clip(prob_ratios, min=clips_lower, max=clips_upper)

                # Actor update
                actor_loss = - (torch.min(prob_ratios * batch_advantages,
                                        clipped_prob_ratios * batch_advantages)).mean()
                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                self.actor_optimiser.step()

                # Critic update
                state_values = self.critic.forward(batch_states).reshape(-1)
                batch_v_trace_targets = batch_v_trace_targets.reshape(-1)
                critic_loss = F.mse_loss(state_values, batch_v_trace_targets)
                self.critic_optimiser.zero_grad()
                critic_loss.backward()
                self.critic_optimiser.step()

        # Update the old actor
        self.old_actor = deepcopy(self.actor)

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
        """Simulate episode using the current policy to estimate returns for ensemble weighting."""
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