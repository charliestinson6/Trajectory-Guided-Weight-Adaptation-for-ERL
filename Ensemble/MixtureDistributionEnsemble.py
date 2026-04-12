import torch
import numpy as np
import matplotlib.pyplot as plt
from Part1.Environments.simple_grid import RandomSimpleGridEnv
from Part1.Algorithms.Ensembles.BaseEnsemble import BaseEnsemble
from Part1.Buffers.ReplayBuffer import ReplayBuffer
import torch.multiprocessing as mp
import random
from copy import deepcopy


class MixtureDistributionEnsemble(BaseEnsemble):
    def __init__(self, env, *agents, fixed_weights=False, uncertainty_temperature=0, boltzmann_temperature=1, alpha=0.5, noisy_model=False, seed=None, env_seed=None):
        super().__init__(env, seed)
        self.env.action_space.seed(seed)
        self.agents = agents
        self.num_agents = len(agents)
        self.fixed_weights = fixed_weights
        self.uncertainty_temperature = uncertainty_temperature
        self.boltzmann_temperature = boltzmann_temperature
        self.alpha = alpha
        self.noisy_model = noisy_model
        self.seed = seed
        self.env_seed = env_seed

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        for agent in self.agents:
            agent.class_name = agent.__class__.__name__

        # For tracking rewards
        self.episode_rewards = []
        self.cumulative_rewards = []

        # Initialise ensemble weights to be equal for all agents
        self.ensemble_weights = np.ones(len(agents)) / len(agents)
        self.ensemble_weights_history = [self.ensemble_weights.tolist()]

        # For tracking simulated returns and their statistics
        self.simulated_returns = [[] for _ in range(self.num_agents)]
        self.moving_averages = np.zeros(self.num_agents)
        self.moving_variances = np.zeros(self.num_agents)
        
        self.moving_averages_history = []
        self.moving_variances_history = []

    def train(self, training_episodes=None):
        """Train the ensemble for a specified number of episodes, updating ensemble weights based on simulated returns."""
        self.ensemble_action_probs = []
        # Initialise obstacle seed tracking
        self.current_obstacle_seed = None

        # ------------------------------------------------------------
        # Determine base seed for environment resets
        # Priority:
        #   1) Environment-defined seed (if explicitly exposed)
        #   2) User-provided env_seed
        #   3) Agent-level seed (fallback)
        #
        # This ensures reproducibility while allowing external control
        # over environment stochasticity when needed.
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
            # Episode-specific seed
            seed = base_seed + episode - 1 if base_seed is not None else None

            # ------------------------------------------------------------
            # Use CURRENT regime for simulation and environment interaction
            # ------------------------------------------------------------
            active_seed = self.current_obstacle_seed if is_dynamic else seed

            # simulate returns for all agents at the start of the episode
            if not self.fixed_weights and episode != 1:
                self.append_simulated_experiences(active_seed)
                self.update_weights()
                self.ensemble_weights_history.append(self.ensemble_weights.tolist())

            # reset environment using current regime
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
            # Train the ensemble
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward

                # Store experience in each agent's replay buffer
                for agent in self.agents:
                    agent.buffer.add(state, action, reward, next_state, done, truncated)

                state = next_state
                
                for agent in self.agents:
                    # Update agents which are updated every step
                    if agent.update_mode == "step" and agent.buffer.size() >= agent.batch_size:
                        agent.update()
                    # Update agents which are updated every rollout
                    elif agent.update_mode == "rollout" and agent.rollout_length is not None and agent.buffer.size() >= agent.rollout_length:
                        agent.update(behaviour_action_probs=self.ensemble_action_probs[-agent.rollout_length:])
            
            # Add episode reward to tracking lists
            self.episode_rewards.append(episode_reward)
            if self.cumulative_rewards:
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + episode_reward)
            else:
                self.cumulative_rewards.append(episode_reward)

            for agent in self.agents:
                # Update agents which are updated at the end of every episode (rollout_length=None)
                if agent.update_mode == "rollout" and agent.rollout_length is None:
                    agent.update(behaviour_action_probs=self.ensemble_action_probs)
                # Decay epsilon for agents that use epsilon-greedy exploration
                if hasattr(agent, 'decay_epsilon'):
                    agent.decay_epsilon(episode)
            
            # Progress printing
            if (episode / training_episodes * 100).is_integer():
                print(f'{int(episode / training_episodes * 100)}% Complete')

    def get_action(self, state):
        """Select an action based on the mixture distribution with the current ensemble weights."""
        # Select agent i with probability self.ensemble_weights[i]
        i = np.random.choice(self.num_agents, p=self.ensemble_weights)
        action = self.agents[i].get_action(state)

        agent_action_probs = np.zeros((self.num_agents, self.env.action_space.n), dtype=np.float32)

        # Calculate action probabilities for each agent in state
        for a in range(self.num_agents):
            agent_action_probs[a] = self.agents[a].return_action_probs(state)

        # Mixture: sum over agents, weighted by ensemble_weights
        ensemble_action_probs = np.sum(self.ensemble_weights[:, None] * agent_action_probs, axis=0)

        # Normalise to ensure it's a probability distribution
        ensemble_action_probs /= np.sum(ensemble_action_probs)

        # Append the probability of the selected action
        self.ensemble_action_probs.append(ensemble_action_probs[action])

        return action
    
    def simulate_agent_returns(self, seed=None):
        """Simulate returns for all agents."""
        #self.env.unwrapped.render_flag = True
        if isinstance(self.env.unwrapped, RandomSimpleGridEnv):
            self._temporarily_adjust_environment_noise(model_noise=True)

        agent_returns = [agent.simulate_returns(seed) for agent in self.agents]

        if isinstance(self.env.unwrapped, RandomSimpleGridEnv):
            self._restore_environment_noise()
        #self.env.unwrapped.render_flag = False
        return agent_returns
    
    def append_simulated_experiences(self, seed=None):
        """Simulate experiences for agents and update expected returns and variances."""
        if hasattr(self.env.unwrapped, 'dynamic_obstacle') and self.env.unwrapped.dynamic_obstacle:
            self.env.unwrapped.trajectory_simulation = True
            agent_returns = self.simulate_agent_returns(seed)
            self.env.unwrapped.trajectory_simulation = False
        else:
            agent_returns = self.simulate_agent_returns(seed)

        for a in range(self.num_agents):
            self.simulated_returns[a].extend(agent_returns[a])
    
    def update_weights(self):
        """Update ensemble weights based on simulated returns using a Boltzmann distribution."""
        latest_returns = [self.simulated_returns[a][-1] for a in range(self.num_agents)]
        mean_return = sum(latest_returns) / self.num_agents

        for a in range(self.num_agents):
            centered_return = latest_returns[a] - mean_return
            delta = centered_return - self.moving_averages[a]
            self.moving_averages[a] += self.alpha * delta
            self.moving_variances[a] = self.alpha * delta**2 + (1 - self.alpha) * self.moving_variances[a]
    
        self.moving_averages_history.append(deepcopy(self.moving_averages).tolist())
        self.moving_variances_history.append(deepcopy(self.moving_variances).tolist())

        scores = (self.moving_averages - self.uncertainty_temperature * np.sqrt(self.moving_variances)) / self.boltzmann_temperature
        max_score = np.max(scores)
        unnormalised_weights = np.exp(scores - max_score)
        self.ensemble_weights = unnormalised_weights / np.sum(unnormalised_weights)

    def _temporarily_adjust_environment_noise(self, model_noise=False):
        """Temporarily adjust the noisy action and observation probabilities."""
        if self.noisy_model:
            if hasattr(self.env, 'noisy_action_prob'):
                if model_noise:
                    self.env.noisy_action_prob += 0.05
                else:
                    self.env.noisy_action_prob -= 0.05
            if hasattr(self.env, 'noisy_observation_prob'):
                if model_noise:
                    self.env.noisy_observation_prob += 0.05
                else:
                    self.env.noisy_observation_prob -= 0.05

    def _restore_environment_noise(self):
        """Restore the original noisy action and observation probabilities."""
        self._temporarily_adjust_environment_noise(model_noise=False)

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
        