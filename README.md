# Trajectory Guided Weight Adaptation for Ensemble Reinforcement Learning

## Overview

The **Dynamic Ensemble** module provides a framework for combining multiple reinforcement learning (RL) agents into a single decision-making system using a **mixture distribution over policies**. This allows the ensemble to adaptively weight agents based on performance, uncertainty, or fixed configurations.

The framework is designed to be **environment-agnostic** and can be used with both custom environments (e.g. grid-based or partially observable mazes) and standard **Gymnasium discrete-action environments** such as `LunarLander-v3`, `CartPole-v1`, and other classic control tasks.

The key component is:

* `MixtureDistributionEnsemble`: Combines multiple agents using either **dynamic weighting** or **fixed weighting**

This README covers:

1. Creating a single agent (DDQN example)
2. Creating a dynamic ensemble
3. Using fixed equal weights
4. Customising the maze environments

---

## 1. Creating a Simple Agent (DDQN Example)

Below are minimal examples of creating and training a **Double Deep Q-Network (DDQN)** agent on both a custom environment and a standard Gymnasium discrete-action environment.

---

### Example 1: Custom Grid Environment

```python
import gymnasium as gym
from Algorithms.DDQN import DDQN

env = gym.make(
    'RandomSimpleGrid-v0',
    rows=8,
    cols=10,
    num_obstacles=20,
    max_steps=1000,
    step_reward=-0.1,
    wall_reward=-2,
    completion_reward=100,
    simple=False,
    partially_observable=False,
    dynamic_obstacle=True,
    update_frequency=50,
    seed=env_seed,
    render_flag=False
)

agent = DDQN(
    env,
    buffer_size=1000,
    batch_size=32,
    hidden_dims=(128, 64),
    lr=0.00025,
    tau=0.01,
    seed=alg_seed,
    env_seed=env_seed
)

agent.train(num_episodes)
```

### Example 2: Gymnasium Discrete Environment (LunarLander)
```python
import gymnasium as gym
from Algorithms.DDQN import DDQN

env = gym.make("LunarLander-v3")

agent = DDQN(
    env,
    buffer_size=20000,
    batch_size=64,
    hidden_dims=(256, 128),
    lr=2.5e-4,
    tau=0.005,
    seed=alg_seed,
    env_seed=env_seed
)

agent.train(num_episodes)
```

---

## 2. Creating a Dynamic Ensemble

The Dynamic Ensemble combines multiple agents and learns **adaptive weights** based on simulated returns.

### Example

```python
import gymnasium as gym
from Algorithms.DDQN import DDQN
from Algorithms.SAC import SAC
from Algorithms.OffPolicyPPO import OffPolicyPPO
from Algorithms.Ensembles.MixtureDistributionEnsemble import MixtureDistributionEnsemble

env = gym.make(
    'RandomSimpleGrid-v0',
    rows=8,
    cols=10,
    num_obstacles=20,
    max_steps=1000,
    step_reward=-0.1,
    wall_reward=-2,
    completion_reward=100,
    simple=False,
    partially_observable=False,
    dynamic_obstacle=True,
    update_frequency=50,
    seed=env_seed,
    render_flag=False
)

ddqn = DDQN(env, buffer_size=1000, batch_size=32, hidden_dims=(128, 64),
            lr=0.00025, tau=0.01, seed=alg_seed, env_seed=env_seed)

ppo = OffPolicyPPO(env,
    actor_hidden_dims=(256, 256, 128),
    critic_hidden_dims=(256, 256, 128),
    batch_size=64,
    actor_lr=0.0005,
    critic_lr=0.0005,
    epochs=7,
    lam=0.95,
    epsilon=0.05,
    rollout_length=None,
    seed=alg_seed,
    env_seed=env_seed
)

sac = SAC(env,
    buffer_size=2500,
    batch_size=64,
    actor_hidden_dims=(256, 128),
    critic_hidden_dims=(256, 128),
    actor_lr=0.00025,
    critic1_lr=0.00025,
    critic2_lr=0.00025,
    log_alpha_lr=1e-5,
    tau=0.001,
    seed=alg_seed,
    env_seed=env_seed
)

ensemble = MixtureDistributionEnsemble(
    env,
    ddqn,
    sac,
    ppo,
    fixed_weights=False,
    uncertainty_temperature=0.25,
    boltzmann_temperature=50,
    alpha=0.25,
    noisy_model=True,
    seed=alg_seed,
    env_seed=env_seed
)

ensemble.train(num_episodes)
```

---

## 3. How the Dynamic Ensemble Works

### Action Selection

At each timestep:

1. Sample an agent index

   * Preferred (if supported):
     $i \sim \mathrm{Categorical}(w)$
   * Fallback:
     i ~ Categorical(w)

2. Use agent (i) to select the action

---

### Learning

* All agents share the same experience stream
* Each agent updates independently using its own algorithm
* The ensemble tracks:

  * Simulated returns
  * Moving averages
  * Moving variances

---

### Weight Updates

Weights are updated using a **Boltzmann distribution over risk-adjusted returns**.

* Preferred (if supported):
  $$
  w_i \propto \exp\left(\frac{\mu_i - \lambda \sigma_i}{T}\right)
  $$

* Fallback (always readable):

```
w_i ∝ exp((μ_i - λ * σ_i) / T)
```

Where:

* μ_i: moving average return
* σ_i: uncertainty (variance)
* λ: uncertainty penalty (`uncertainty_temperature`)
* T: softmax temperature (`boltzmann_temperature`)

---

## 4. Using Fixed Weights

To disable adaptive weighting:

```python
ensemble = MixtureDistributionEnsemble(
    env,
    ddqn,
    sac,
    ppo,
    fixed_weights=True
)
```

### Behaviour

* Weights remain constant:

```python
self.ensemble_weights = np.ones(num_agents) / num_agents
```

* No simulation or weight updates occur
* Equivalent to a **static mixture policy**

---

## 5. Customising the Maze Environment

The `RandomSimpleGrid-v0` environment is highly configurable.

### Core Parameters

```python
env = gym.make(
    'RandomSimpleGrid-v0',
    rows=8,
    cols=10,
    num_obstacles=20,
    max_steps=1000,
    step_reward=-0.1,
    wall_reward=-2,
    completion_reward=100,
)
```

---

### Environment Modes

Only **one** of the following can be `True`:

| Mode                        | Description                      |
| --------------------------- | -------------------------------- |
| `simple=True`               | Fully observable grid            |
| `partially_observable=True` | Noisy local observations (POMDP) |
| `dynamic_obstacle=True`     | Maze changes over time           |

---

### Dynamic Obstacles

```python
dynamic_obstacle=True,
update_frequency=50
```

* Maze layout changes every `update_frequency` episodes
* Controlled via environment seed
* Ensemble maintains consistent obstacle regimes within episodes

---

### Noise Configuration

```python
noisy_actions=True,
noisy_action_prob=0.2,
noisy_observations=True,
noisy_observation_prob=0.1
```

* Action noise: random action override
* Observation noise: corrupted wall detection
* Useful for robustness and uncertainty modelling

---

### Reward Structure

```python
step_reward = -0.1        # per step penalty
wall_reward = -2          # hitting wall
completion_reward = 100   # reaching goal
```

---

### Custom Maze Layout

```python
custom_map = [
    "0000",
    "0101",
    "0001",
    "1000"
]
```

To use this, extend the environment constructor to accept a custom map.

---

### Start and Goal Control

```python
state, _ = env.reset(options={
    "starts_xy": (0, 0),
    "goals_xy": (7, 9)
})
```

---

## 6. Reproducibility

The system uses **three levels of seeding**:

1. `env_seed` → controls environment stochasticity
2. `seed` → controls agent randomness
3. Episode-based seed progression → ensures reproducible trajectories

Dynamic environments maintain:

* Fixed obstacle regimes within update windows
* Controlled transitions across episodes

