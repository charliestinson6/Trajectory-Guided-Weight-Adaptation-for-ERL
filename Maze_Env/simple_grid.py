from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import random
import torch

MAPS = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}

class RandomSimpleGridEnv(Env):
    """
    Simple Grid Environment

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when the agents reaches its goal.

    To initialise the grid, the user must decide where to put the walls on the grid. This can be done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000", 
     "0101", 
     "0001", 
     "1000"]

    Assume the environment is a grid of size (nrow, ncol). A state s of the environment is an elemente of gym.spaces.Discete(nrow*ncol), i.e. an integer between 0 and nrow * ncol - 1. Assume nrow=ncol=5 and s=10, to compute the (x,y) coordinates of s on the grid the following formula are used: x = s // ncol  and y = s % ncol.
     
    The user can also decide the starting and goal positions of the agent. This can be done by through the `options` dictionary in the `reset` method. The user can specify the starting and goal positions by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type int (s) or tuple (x,y) and represent the agent starting and goal positions respectively. 
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(self,   
        rows: int,
        cols: int,
        num_obstacles: int=16,
        simple: bool = True,
        partially_observable: bool = False,
        dynamic_obstacle: bool = False,
        update_frequency: int = 1,
        noisy_actions: bool = True,
        noisy_action_prob: float = 0.2,
        noisy_observations: bool = True,
        noisy_observation_prob: float = 0.1,
        max_steps: int = 1000,
        step_reward: int = -0.1,
        wall_reward: int = -2,
        completion_reward: int = 100,
        render_flag: bool = False,
        render_mode: str | None = 'human',
        seed: int | None = None
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        agent_color: str
            Color of the agent. The available colors are: red, green, blue, purple, yellow, grey and black. Note that the goal cell will have the same color.
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. 
            An example of a 4x4 map is the following:
            ["0000",
             "0101", 
             "0001",
             "1000"]
        """

        self.seed = seed
        self.nrow = rows
        self.ncol = cols
        self.num_obstacles = num_obstacles
        
        self.noisy_actions = noisy_actions
        self.noisy_action_prob = noisy_action_prob
        if not self.noisy_actions:
            self.noisy_action_prob = 0.0

        self.noisy_observations = noisy_observations
        self.noisy_observation_prob = noisy_observation_prob
        if not self.noisy_observations:
            self.noisy_observation_prob = 0.0

        self.max_steps = max_steps
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.completion_reward = completion_reward

        self.simple = simple
        self.partially_observable = partially_observable
        self.dynamic_obstacle = dynamic_obstacle
        self.update_frequency = update_frequency
        if (self.simple and self.partially_observable) or (self.simple and self.dynamic_obstacle) or (self.partially_observable and self.dynamic_obstacle):
             raise ValueError('Only 1 of simple, partially_observable and dynamic_obstacle can be True.')

        self.render_flag = render_flag

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # Env confinguration
        if self.simple:
            obstacle_map, _, _ = self.generate_random_maze(rows, cols, self.num_obstacles)
            self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
            self.obstacles_array = self.obstacles.reshape(self.nrow * self.ncol)
            self.belief_state = np.zeros(self.nrow * self.ncol)
        elif self.partially_observable:
            obstacle_map, _, _ = self.generate_random_maze(rows, cols, self.num_obstacles)
            self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
            self.obstacles_array = self.obstacles.reshape(self.nrow * self.ncol)
            # Initialise belief states
            free_cells = np.count_nonzero(self.obstacles_array == 0)
            self.belief_state = np.array([1/free_cells if self.obstacles_array[i] == 0 else 0 for i in range(self.obstacles_array.shape[0])])
        else:
            # Initialise belief states
            self.belief_state = np.zeros(self.nrow * self.ncol)

        self.action_space = spaces.Discrete(len(self.MOVES))
        if self.simple:
            self.observation_space = spaces.MultiBinary(n=self.nrow * self.ncol)
        elif self.partially_observable:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.nrow * self.ncol,), dtype=np.float32)
        else:
            self.observation_space = spaces.MultiBinary(n=self.nrow * self.ncol * 2)

        # Rendering configuration
        self.fig = None

        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']

    def generate_random_maze(self, rows, cols, num_obstacles):
        # Initialise the maze with all walls
        maze = [[1 for _ in range(cols)] for _ in range(rows)]
        if not self.partially_observable: # Set start and end to top-left and bottom-right grid positions
            start = self.to_xy(0)
            end = self.to_xy(rows * cols - 1)
        else: # Set end to bottom-right grid position and start randomly
            while True:
                start_index = np.random.randint(0, rows * cols)
                if start_index != rows * cols - 1:
                    break
            start = self.to_xy(start_index)
            end = self.to_xy(rows * cols - 1)

        maze[start[0]][start[1]] = 0  # Ensure start isn't a wall
        maze[end[0]][end[1]] = 0  # Ensure end isn't a wall
        frontiers = self.in_bounds_states(start[0], start[1])  # Initialise frontiers with the start position

        # While there are frontiers
        while frontiers:
            random_index = random.randint(0, len(frontiers) - 1)
            x, y = frontiers.pop(random_index)

            # Count the number of neighboring paths
            neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0]

            if len(valid_neighbors) == 1:  # Add path if exactly one neighbor is already a path
                maze[x][y] = 0

                # Add neighbors of this cell to the frontier
                for nx, ny in neighbors:
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                        frontiers.append((nx, ny))

        # Perform DFS or BFS to check if the maze is connected
        def is_reachable(start, end):
            stack = [start]
            visited = set()
            while stack:
                x, y = stack.pop()
                if (x, y) == end:
                    return True
                for nx, ny in [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]:
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            return False

        # Get all coordinates of existing obstacle cells
        obstacle_cells = [(r, c) for r in range(rows) for c in range(cols) if maze[r][c] == 1]

        # Randomly select cells to retain as obstacles
        if num_obstacles < len(obstacle_cells):
            # Use class RNG instance instead of global random
            retained_obstacles = random.sample(obstacle_cells, k=num_obstacles)
            
            # Set all other obstacle cells to 0
            for r, c in obstacle_cells:
                if (r, c) not in retained_obstacles:
                    maze[r][c] = 0
        else:
            retained_obstacles = obstacle_cells

        # If the finish is not reachable, move obstacles until it is reachable
        while not is_reachable(start, end):
            for r, c in retained_obstacles:
                if maze[r][c] == 1:
                    # Find a free cell to move the obstacle to
                    free_cells = [(i, j) for i in range(rows) for j in range(cols) if (i, j) not in (start, end) and maze[i][j] == 0]
                    if free_cells:
                        new_r, new_c = random.choice(free_cells)
                        maze[r][c] = 0
                        maze[new_r][new_c] = 1
                        retained_obstacles.remove((r, c))
                        retained_obstacles.append((new_r, new_c))
                        break

        # Convert the maze to a list of strings
        maze = [''.join(str(maze[i][j]) for j in range(cols)) for i in range(rows)]
        return maze, start, end

    def reset(
            self, 
            seed: int | None = None,
            render_flag: bool = True,
            options: dict = dict()
        ) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        
        if self.simple:
            start = (0, 0)
            end = (self.nrow - 1, self.ncol - 1)
        elif self.partially_observable:
            end = (self.nrow - 1, self.ncol - 1)
            while True:
                start_index = np.random.randint(0, self.nrow * self.ncol - 1)
                start = self.to_xy(start_index)
                if self.is_free(*start) and self.is_reachable_by_path(start, end):
                    break
        else:
            start = (0, 0)
            end = (self.nrow - 1, self.ncol - 1)
            obstacle_map, _, _ = self.generate_random_maze(self.nrow, self.ncol, self.num_obstacles)
            self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
            self.obstacles_array = self.obstacles.reshape(self.nrow * self.ncol)

        # parse options
        self.start_xy = start
        self.goal_xy = end

        # Rendering configuration
        self.fig = None
        self.render_mode = self.render_mode
        self.fps = self.metadata['render_fps']

        # initialise internal vars
        self.agent_xy = self.start_xy

        # Update state
        self.agent_x = self.agent_xy[0]       
        self.agent_y = self.agent_xy[1]
        self.agent_state = np.concatenate((self.obstacles_array, 
            np.array([1 if i == self.agent_x * self.ncol + self.agent_y else 0 for i in range(self.nrow * self.ncol)]))).astype(np.float32)

        if self.partially_observable:
            # Re-initialise belief to uniform over free cells
            self.obstacles_array = self.obstacles.reshape(self.nrow * self.ncol)
            free_cells = np.count_nonzero(self.obstacles_array == 0)
            self.belief_state = np.array([
                1/free_cells if self.obstacles_array[i] == 0 else 0
                for i in range(self.obstacles_array.shape[0])
            ]).astype(np.float32)

            north_wall = int(not(self.is_in_bounds(self.agent_xy[0], self.agent_xy[1] - 1) and self.is_free(self.agent_xy[0], self.agent_xy[1] - 1)))
            east_wall  = int(not(self.is_in_bounds(self.agent_xy[0] + 1, self.agent_xy[1]) and self.is_free(self.agent_xy[0] + 1, self.agent_xy[1])))
            south_wall = int(not(self.is_in_bounds(self.agent_xy[0], self.agent_xy[1] + 1) and self.is_free(self.agent_xy[0], self.agent_xy[1] + 1)))
            west_wall  = int(not(self.is_in_bounds(self.agent_xy[0] - 1, self.agent_xy[1]) and self.is_free(self.agent_xy[0] - 1, self.agent_xy[1])))
            walls = [north_wall, east_wall, south_wall, west_wall]
            
            if self.noisy_observations:
                # Generate initial noisy observation of adjacent walls
                obs_probs = torch.tensor([1 - self.noisy_observation_prob])
                bernoulli_dist = torch.distributions.Bernoulli(obs_probs)

                for i in range(len(walls)):
                    if bernoulli_dist.sample() == 0:
                        walls[i] = 1 - walls[i]

            self.observation = np.array(walls)

        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # Check integrity
        self.integrity_checks()

        #if self.render_mode == "human":
        if self.render_flag is True:
            self.render()

        return self.get_obs(), self.get_info()
    
    def step(self, action: int, render_flag=True):
        """
        Take a step in the environment.
        """
        self.agent_target_action = action
        if self.noisy_actions:
            # Define probabilities for the Bernoulli distribution to add noise to the action
            action_probs = torch.tensor([1 - self.noisy_action_prob])
            # Create a Bernoulli distribution
            bernoulli_dist = torch.distributions.Bernoulli(action_probs)
            if bernoulli_dist.sample() == 0:
                # Select random action
                action = self.action_space.sample()

        #assert action in self.action_space
        self.agent_action = action

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)

        if self.partially_observable:
            # Observation is 1 if there is a wall in the direction, 0 otherwise
            
            # Update observation
            north_wall = int(not(self.is_in_bounds(self.agent_xy[0], self.agent_xy[1] - 1) and self.is_free(self.agent_xy[0], self.agent_xy[1] - 1)))
            east_wall = int(not(self.is_in_bounds(self.agent_xy[0] + 1, self.agent_xy[1]) and self.is_free(self.agent_xy[0] + 1, self.agent_xy[1])))
            south_wall = int(not(self.is_in_bounds(self.agent_xy[0], self.agent_xy[1] + 1) and self.is_free(self.agent_xy[0], self.agent_xy[1] + 1)))
            west_wall = int(not(self.is_in_bounds(self.agent_xy[0] - 1, self.agent_xy[1]) and self.is_free(self.agent_xy[0] - 1, self.agent_xy[1])))
            walls = [north_wall, east_wall, south_wall, west_wall]

            if self.noisy_observations:
                for i in range(len(walls)):
                    if bernoulli_dist.sample() == 0:
                        walls[i] = 1 - walls[i]

            self.observation = np.array(walls)

            self.update_beliefs()

        self.done = self.on_goal()

        self.n_iter += 1
        if self.n_iter >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # if self.render_mode == "human":
        if self.render_flag is True:
            self.render()

        return self.get_obs(), self.reward, self.done, truncated, self.get_info()
    
    def update_beliefs(self):
        """
        Update belief state using Bayes filter: b'(s') ∝ P(o | s') * sum_s P(s' | s, a) * b(s)
        """

        # --- 1. Transition model: P(s' | s, a) ---
        transition_probs = np.zeros((self.nrow * self.ncol, self.nrow * self.ncol))
        for s in range(self.nrow * self.ncol):
            if self.obstacles_array[s] == 1:
                continue
            x, y = self.to_xy(s)
            for a in range(len(self.MOVES)):
                dx, dy = self.MOVES[a]
                nx, ny = x + dx, y + dy
                ns = self.to_pos(nx, ny) if self.is_in_bounds(nx, ny) and self.is_free(nx, ny) else s
                if ns != s:
                    if not self.noisy_actions:
                        prob = 1.0 if a == self.agent_target_action else 0.0
                    else:
                        prob = (1 - self.noisy_action_prob) if a == self.agent_target_action else self.noisy_action_prob / (len(self.MOVES) - 1)
                    transition_probs[s, ns] += prob

        # --- 2. Observation model: P(o | s') ---
        observation_model = np.zeros(self.nrow * self.ncol)
        for s in range(self.nrow * self.ncol):
            if self.obstacles_array[s] == 1:
                continue
            x, y = self.to_xy(s)

            true_walls = [
                int(not (self.is_in_bounds(x, y - 1) and self.is_free(x, y - 1))),  # N
                int(not (self.is_in_bounds(x + 1, y) and self.is_free(x + 1, y))),  # E
                int(not (self.is_in_bounds(x, y + 1) and self.is_free(x, y + 1))),  # S
                int(not (self.is_in_bounds(x - 1, y) and self.is_free(x - 1, y)))   # W
            ]

            if not self.noisy_observations:
                # Deterministic observation model
                prob = 1.0 if np.array_equal(self.observation, true_walls) else 0.0
            else:
                prob = 1.0
                for i in range(4):
                    if self.observation[i] == true_walls[i]:
                        prob *= 1 - self.noisy_observation_prob
                    else:
                        prob *= self.noisy_observation_prob
            observation_model[s] = prob

        # --- 3. Belief update ---
        unnormalised_belief = np.zeros(self.nrow * self.ncol)
        for s_prime in range(self.nrow * self.ncol):
            if self.obstacles_array[s_prime] == 1:
                continue
            unnormalised_belief[s_prime] = observation_model[s_prime] * np.dot(transition_probs[:, s_prime], self.belief_state)

        self.belief_state = (unnormalised_belief / (np.sum(unnormalised_belief) + 1e-8)).astype(np.float32)

    
    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "010", "011]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        elif isinstance(obstacle_map, str):
            map_str = MAPS[obstacle_map]
            map_str = np.asarray(map_str, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        else:
            raise ValueError(f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}.")
        
    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f'Allowed types for `{state_name}` are int or tuple.')
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(f'Key `{state_name}` not found in `options`. Random sampling a valid value for it:')
            logger.info(f'...`{state_name}` has value: {state}')
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.observation_space.sample()
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.observation_space.sample()
            pos_xy = self.to_xy(state)
        return pos_xy
    
    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."
        
    def to_simple_state(self, row: int, col: int):
        """
        Transform a (row, col) point to a state in the observation space for simple maze.
        -- This is just the agent position, not the maze obstacles or start/goal
        """
        # Update state representaiton
        self.agent_x = row
        self.agent_y = col
        self.agent_state = np.array([1 if i == self.agent_x * self.ncol + self.agent_y else 0 for i in range(self.nrow * self.ncol)])
        return self.agent_state
        
    def to_full_state(self, row: int, col: int):
        """
        Transform a (row, col) point to a state in the observation space.
        -- UPDATE state will now be maze and position concatenated
        """
        # Update state representaiton
        self.agent_x = self.agent_xy[0]
        self.agent_y = self.agent_xy[1]
        self.agent_state = np.concatenate((self.obstacles_array, 
            np.array([1 if i == self.agent_x * self.ncol + self.agent_y else 0 for i in range(self.nrow * self.ncol)])))
        return self.agent_state

    def to_pos(self, row: int, col: int) -> int:
        """
        Converts (row, col) to flat index in grid for belief state, obstacle map, etc.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == self.FREE
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol
    
    def target_state_action_counts(self, start_row: int, start_col: int, target_row: int, target_col: int) -> bool:
        """
        Counts the number of possible actions that can lead the agent to the target state from start state.
        """
        action_count = 0
        for action in self.MOVES.keys():
            dx, dy = self.MOVES[action]
            end_row = start_row + dx
            end_col = start_col + dy
            if end_row == target_row and end_col == target_col:
                action_count += 1
        return action_count
    
    def is_reachable(self, start_row: int, start_col: int, target_row: int, target_col: int) -> bool:
        """
        Check if the target state is reachable from start state.
        """
        for action in self.MOVES.keys():
            dx, dy = self.MOVES[action]
            end_row = start_row + dx
            end_col = start_col + dy
            if end_row == target_row and end_col == target_col:
                return True
        return False
    
    def is_reachable_by_path(self, start, end):
        stack = [start]
        visited = set()
        while stack:
            x, y = stack.pop()
            if (x, y) == end:
                return True
            for nx, ny in [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]:
                #ns = self.to_pos(nx, ny) if self.is_in_bounds(nx, ny) and self.is_free(nx, ny) else s
                if self.is_in_bounds(nx, ny) and self.is_free(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        return False

    def in_bounds_states(self, start_row: int, start_col: int) -> bool:
        """
        Return the in bounds target states from start state.
        """
        in_bounds_states = []
        for action in self.MOVES.keys():
            dx, dy = self.MOVES[action]
            end_row = start_row + dx
            end_col = start_col + dy
            if self.is_in_bounds(end_row, end_col):
                in_bounds_states.append((end_row, end_col))
        return in_bounds_states

    def is_reachable_by_action(self, start_row: int, start_col: int, target_row: int, target_col: int, action: int) -> bool:
        """Check if the target state is reachable from the start state using the given action."""
        if not self.is_in_bounds(target_row, target_col) or not self.is_free(target_row, target_col):
            return False  # The target is out of bounds or a wall

        if action not in self.MOVES:
            return False  # Invalid action

        dx, dy = self.MOVES[action]
        expected_row = start_row + dx
        expected_col = start_col + dy

        return expected_row == target_row and expected_col == target_col  # True if action leads exactly to target

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return self.wall_reward
        elif not self.is_free(x, y):
            return self.wall_reward
        elif (x, y) == self.goal_xy:
            return self.completion_reward
        else:
            return self.step_reward

    def get_obs(self):
        if self.simple:
            # Return simple state representation
            return self.to_simple_state(*self.agent_xy)
        if self.partially_observable:
            # Return belief state in POMDP setting
            return self.belief_state.copy()
        else:
            # Return actual state index in fully observable MDP
            return self.to_full_state(*self.agent_xy)

    
    def get_info(self) -> dict:
        return {
            'agent_xy': self.agent_xy,
            'n_iter': self.n_iter,
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
        
        elif self.render_mode == "ansi":
            s = f"{self.n_iter},{self.agent_xy[0]},{self.agent_xy[1]},{self.reward},{self.done},{self.agent_action}\n"
            #print(s)
            return s

        elif self.render_mode == "rgb_array":
            self.render_frame()
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
        elif self.render_mode == "human":
            self.render_frame()
            plt.pause(1/self.fps)
            return None
        
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self):
        if self.fig is None:
            self.render_initial_frame()
            self.fig.canvas.mpl_connect('close_event', self.close)
        else:
            self.update_agent_patch()
        self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward}")
    
    def create_agent_patch(self):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        return mpl.patches.Circle(
            (self.agent_xy[1]+.5, self.agent_xy[0]+.5), 
            0.3, 
            facecolor='blue', 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        @NOTE: If agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) as center to the Circle patch.
        """
        self.agent_patch.center = (self.agent_xy[1]+.5, self.agent_xy[0]+.5)
        return None
    
    def render_initial_frame(self):
        """
        Render the initial frame.

        @NOTE: 0: free cell (white), 1: obstacle (black), 2: start (red), 3: goal (green)
        """
        data = self.obstacles.copy()
        data[self.start_xy] = 2
        data[self.goal_xy] = 3

        colors = ['white', 'black', 'red', 'green']
        bounds=[i-0.1 for i in [0, 1, 2, 3, 4]]

        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()
        fig, ax = plt.subplots(tight_layout=True)
        self.fig = fig
        self.ax = ax

        #ax.grid(axis='both', color='#D3D3D3', linewidth=2) 
        ax.grid(axis='both', color='k', linewidth=1.3) 
        ax.set_xticks(np.arange(0, data.shape[1], 1))  # correct grid sizes
        ax.set_yticks(np.arange(0, data.shape[0], 1))
        ax.tick_params(
            bottom=False, 
            top=False, 
            left=False, 
            right=False, 
            labelbottom=False, 
            labelleft=False
        ) 

        # draw the grid
        ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            extent=[0, data.shape[1], data.shape[0], 0],
            interpolation='none'
        )

        # Create white holes on start and goal positions
        for pos in [self.start_xy, self.goal_xy]:
            wp = self.create_white_patch(*pos)
            ax.add_patch(wp)

        # Create agent patch in start position
        self.agent_patch = self.create_agent_patch()
        ax.add_patch(self.agent_patch)

        return None

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y+.5, x+.5), 
            0.4, 
            color='white', 
            fill=True, 
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        plt.close(self.fig)
        sys.exit()

# Register the environment at the module level (outside the class)
register(
    id='RandomSimpleGrid-v0',
    entry_point='IJCNN2026.Maze_Env.simple_grid:RandomSimpleGridEnv',  # Module path to your environment class
    kwargs={},  # Default arguments can be passed here
)