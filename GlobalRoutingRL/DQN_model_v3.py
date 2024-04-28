# Importing required packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from gym import spaces
import numpy as np
import MST as tree
from collections import deque
from RoutingEnv_v2 import RoutingEnv_v2 as RoutingEnv
from RoutingEnv_v2 import load_input_file, prepareTwoPinList_allNet, evaluation

# Neural network model for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 64]):
        super(DQN, self).__init__()
        layers = []
        input_size = state_size
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Add output layer
        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Replay buffer to store transitions
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01, env=None):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.gamma = gamma  # Discount factor
        self.learning_rate = lr
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.model = DQN(state_size, action_size)  # Main DQN model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.env = env

    def act(self, state):
        possible_actions,action_list= self.env.get_possible_actions() # Get possible actions from the environment in a list of True/False

        if np.random.rand() <= self.exploration_rate:
            # return a possible random action in possible_actions where value is True
            return random.choice([i for i, x in enumerate(possible_actions) if x])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            mask_ = np.isneginf(state_tensor)
            state_tensor[mask_] = -10
            # assert (state_tensor >=-50).all()
            
            q_values = self.model(state_tensor)
            # print("q_values: ", q_values)
            mask = torch.tensor(possible_actions, dtype=torch.bool)
            q_values[~mask] = -float('inf')
            return torch.argmax(q_values).item()
            

    def replay(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample()
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.bool)

            mask = np.isneginf(next_state_tensor)
            next_state_tensor[mask] = -10
            
            target = reward_tensor
            if not done_tensor:
                target += self.gamma * torch.max(self.model(next_state_tensor))

            mask = np.isneginf(state_tensor)
            state_tensor[mask] = -10
            q_values = self.model(state_tensor)
            loss = nn.functional.smooth_l1_loss(q_values[action], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)



import numpy as np  # For array operations
# Import other necessary libraries or modules
# import ...

def solve_routing_with_dqn(input_file_path):
    state_size = 18  # Given state size
    action_size = 6  # 4 directions and 2 layer transitions

    # Load the input file to initialize environment variables
    grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name = load_input_file(input_file_path)

    env = RoutingEnv(
        grid_size=grid_size,
        vertical_capacity=vertical_capacity,
        horizontal_capacity=horizontal_capacity,
        minimum_width=minimum_width,
        minimum_spacing=minimum_spacing,
        via_spacing=via_spacing,
        grid_origin=grid_origin,
        grid_dimensions=grid_dimensions,
        adjustments=adjustments,
    )
    
    # Initialize the DQN agent
    agent = DQNAgent(
        state_size,
        action_size,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.5,
        env=env,
    )
    
    # Prepare data structures for routing
    nets_mst = []
    pinList_allNet = prepareTwoPinList_allNet(nets_scaled)
    for net_id, pin_pairs in pinList_allNet.items():
        nets_mst.append(tree.generateMST(pin_pairs))
    
    # Initialize the first net and pin pair
    net_index = 0
    net_pin_pairs = nets_mst[net_index]
    pin_pair_index = 0

    env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)

    # Define episode parameters
    episodes_per_pair = 1  # Number of episodes per 2-pin pair
    episodes_success = 0  # Track successful episodes
    total_rewards = 0  # Track total rewards for the routing process

    done = False  # Flag to indicate if the current episode is finished
    num_total_steps = 0  # Track the number of steps in the current episode

    # Loop to solve the routing problem with episodes
    while True:
        num_total_steps += 1
        
        # DQN agent chooses an action
        current_state = env.state
        action = agent.act(current_state)  # Exploration-exploitation strategy
        next_state, reward, done, info = env.step(action)  # Step in the environment
        
        # Store the transition in the replay buffer and replay for training
        agent.buffer.add(current_state, action, reward, next_state, done)
        agent.replay()  # Experience replay for training
        agent.update_exploration_rate()  # Decay exploration rate
        
        total_rewards += reward  # Accumulate rewards for this episode

        if done:
            # Handling end of a 2-pin pair
            if pin_pair_index < len(net_pin_pairs) - 1:
                if reward > 0:
                    if episodes_success < episodes_per_pair:
                        episodes_success += 1  # Increment successful episodes count
                        env.init_new_pair_state(pin_pair_index)  # Reset for the same pair
                    else:
                        pin_pair_index += 1  # Move to the next pair
                        env.update_env_info(Finish_pair=True)  # Update env info
                        env.init_new_pair_state(pin_pair_index)  # Initialize next pair
                        episodes_success = 0  # Reset success count
                    done = False
                else:
                    env.init_new_pair_state(pin_pair_index)
                    done = False
            elif pin_pair_index == len(net_pin_pairs) - 1:
                if reward > 0:
                    if episodes_success < episodes_per_pair:
                        episodes_success += 1  # Continue with the same pair
                        env.init_new_pair_state(pin_pair_index)
                        done = False
                    else:
                        # If all episodes for this pair are completed, move to the next net
                        net_index += 1
                        episodes_success = 0
                        if net_index < len(nets_mst):
                            # If there are more nets to solve
                            net_pin_pairs = nets_mst[net_index]
                            pin_pair_index = 0
                            env.update_env_info(Finish_pair=True, Finish_net=True)
                            env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)
                            done = False
                        else:
                            # All nets are completed
                            env.update_env_info(Finish_pair=True, Finish_net=True)
                            print("Finish all nets")
                            print("Total rewards:", total_rewards)
                            print("Total steps:", num_total_steps)
                            break
                else:
                    env.init_new_pair_state(pin_pair_index)
                    done = False

    # Calculate total wirelength and overflow
    total_wirelength = 0
    for net_index in range(len(nets_mst)):
        total_wirelength += len(env.nets_visited[net_index]) - 1

    mask_h = (env.capacity_info_h < 0) & (env.capacity_info_h > -10)
    mask_v = (env.capacity_info_v < 0) & (env.capacity_info_v > -10)
    overflow = np.sum(env.capacity_info_h[mask_h]) + np.sum(env.capacity_info_v[mask_v])

    total_congestion, min_capacity, total_wire_length = evaluation(env)


    
    # Return the calculated results and total rewards
    return total_congestion, min_capacity, total_wire_length


# Main block that calls the function
if __name__ == '__main__':
    file_path = 'benchmark/test_benchmark_1.gr'
    ttotal_congestion, min_capacity, total_wire_length = solve_routing_with_dqn(file_path)
    print("Total Congestion:", ttotal_congestion)
    print("Min Capacity:", min_capacity)
    print("Total Wire Length:", total_wire_length)
    



