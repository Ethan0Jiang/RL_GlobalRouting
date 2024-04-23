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
from RoutingEnv import RoutingEnv
from RoutingEnv import load_input_file, prepareTwoPinList_allNet

# Neural network model for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 64]):
        super(DQN, self).__init__()
        layers = []
        input_size = state_size
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
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
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
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

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.choice(self.action_size)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def replay(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample()
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.bool)

            target = reward_tensor
            if not done_tensor:
                target += self.gamma * torch.max(self.model(next_state_tensor))

            q_values = self.model(state_tensor)
            loss = nn.functional.smooth_l1_loss(q_values[action], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)



if __name__ == '__main__':
    # Initialize DQN agent with appropriate parameters
    state_size = 18  # Given state size
    action_size = 6  # 4 directions and 2 layer transitions

    agent = DQNAgent(state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.99)


    grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name = load_input_file(input_file_path='benchmark_reduced/test_benchmark_1.gr')

    env = RoutingEnv(grid_size=grid_size, vertical_capacity=vertical_capacity, horizontal_capacity=horizontal_capacity, 
                     minimum_width=minimum_width, minimum_spacing=minimum_spacing, via_spacing=via_spacing, 
                     grid_origin=grid_origin, grid_dimensions=grid_dimensions, adjustments=adjustments)
    
    nets_mst = []
    pinList_allNet = prepareTwoPinList_allNet(nets_scaled)
    for net_id, pin_pairs in pinList_allNet.items():
        nets_mst.append(tree.generateMST(pin_pairs))

    # Initialize the first net and pin pair
    net_index = 0
    net_pin_pairs = nets_mst[net_index]
    pin_pair_index = 0

    # Initialize the environment with the first net and pin pair
    env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)

    # Define episode parameters
    episodes_per_pair = 10  # Number of episodes per 2-pin pair
    episodes_success = 0  # Track successful episodes
    total_rewards = 0  # Track total rewards for the routing process

    done = False  # Flag to indicate if the current episode is finished

    # Loop to solve the routing problem with episodes
    while True:
        # DQN Agent chooses an action
        current_state = env.state
        action = agent.act(current_state)  # Exploration-exploitation strategy
        next_state, reward, done, info = env.step(action)  # Step in the environment

        # Store the transition in the replay buffer
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
                        print("Success: ", episodes_success)
                    else:
                        # If all episodes for this pair are completed
                        pin_pair_index += 1  # Move to the next pair
                        env.update_env_info(Finish_pair=True)  # Update env info
                        env.init_new_pair_state(pin_pair_index)  # Initialize next pair
                        episodes_success = 0  # Reset success count
                    done = False
                else:
                    # If the routing failed, retry the same pair
                    env.init_new_pair_state(pin_pair_index)
                    done = False
            elif pin_pair_index == len(net_pin_pairs) - 1:
                # Handling the last pair in the net
                if reward > 0:
                    if episodes_success < episodes_per_pair:
                        episodes_success += 1
                        env.init_new_pair_state(pin_pair_index)  # Reset for this pair
                        done = False
                        print("Success: ", episodes_success)
                    else:
                        # If all episodes for this pair are completed
                        net_index += 1  # Move to the next net
                        episodes_success = 0  # Reset success count
                        if net_index < len(nets_mst):
                            # If there are more nets to solve
                            net_pin_pairs = nets_mst[net_index]  # Get the new net
                            pin_pair_index = 0  # Reset to the first pair
                            env.update_env_info(Finish_pair=True, Finish_net=True)
                            env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)
                            done = False
                        else:
                            # All nets are completed
                            env.update_env_info(Finish_pair=True, Finish_net=True)
                            print("Finish all nets")
                            break
                else:
                    # If failed to finish the last pair
                    env.init_new_pair_state(pin_pair_index)
                    done = False
