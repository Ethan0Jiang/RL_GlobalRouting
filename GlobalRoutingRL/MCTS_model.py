import random
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import gym
from gym import spaces
import MST as tree
from RoutingEnv_v2 import RoutingEnv_v2, load_input_file, prepareTwoPinList_allNet, evaluation



class MCTSNode:
    def __init__(self, state, parent=None, action=None, env_copy=None, reward=0, done=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0  # Accumulated value (sum of rewards)
        self.action = action # The action that led to this node
        self.env_copy = env_copy  # Store the environment copy
        self.reward = reward # the action's reward led to this node
        self.done = done

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0 or self.done

    def uct_value(self, exploration_constant):
        if self.done == True and self.reward < 0:
            return float("-inf")
        elif self.visit_count == 0:
            return float("inf")  # Unvisited nodes have infinite exploration value
        return (self.total_value / self.visit_count) + exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
    
    def uct_value_decision(self, exploration_constant):
        return (self.total_value / self.visit_count) + exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )


class MCTSTree:
    def __init__(self, exploration_constant=1.4):
        self.exploration_constant = exploration_constant
        self.root = None

    def set_root(self, state, env_copy):
        self.root = MCTSNode(state=state, env_copy=env_copy)

    def select_best_child(self, node):
        return max(
            node.children,
            key=lambda child: child.uct_value(self.exploration_constant),
        )
    
    def select_best_child_decision(self, node):
        return max(
            node.children,
            key=lambda child: child.uct_value_decision(self.exploration_constant),
        )
    
    def select_best_subnode(self, node):
        return max(
            node.children,
            key=lambda child: child.total_value,
        )

    def expand(self, node):
        original_env = node.env_copy  # Keep the original environment
        _, possible_actions = original_env.get_possible_actions()  # Get valid actions
        
        for action in possible_actions:
            if action is not None:
                env_copy = copy.deepcopy(original_env)  # Create a new deep copy for each iteration
                next_state, reward, done, _ = env_copy.step(action)  # Simulate the action
                child_node = MCTSNode(
                    state=next_state, parent=node, action=action, env_copy=env_copy, reward=reward, done=done
                )
                node.add_child(child_node)

    def backpropagate(self, node, reward):
        current_node = node
        while current_node is not None:
            current_node.visit_count += 1
            current_node.total_value += reward
            current_node = current_node.parent


class MCTS:
    def __init__(self, exploration_constant=1.4, num_simulations=100):
        self.tree = MCTSTree(exploration_constant=exploration_constant)
        self.num_simulations = num_simulations

    def perform_mcts(self, env, initial_state):
        self.tree.set_root(state=initial_state, env_copy=copy.deepcopy(env))  # Set root with env copy

        # Perform simulations
        for _ in range(self.num_simulations):
            current_node = self.tree.root
            while len(current_node.children)>0:
                current_node = self.tree.select_best_child(current_node)

            if current_node.visit_count == 0 and not current_node.done:
                self.tree.expand(current_node)  # Expand with the stored env copy

            if not current_node.done:
                reward = self.simulate(current_node)  # Use the environment copy in simulation
                self.tree.backpropagate(current_node, reward)

        # Retrieve the best sequence of actions
        return self.get_best_action_sequence(10)  # Return the top N action sequence

    def simulate(self, node):
        env_copy = copy.deepcopy(node.env_copy)
        current_state = node.state

        total_reward = 0.0
        done = node.done
        while not done:
            _, possible_actions = env_copy.get_possible_actions()
            # move greedy to the target
            # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z
            distance = current_state[3:6] # in x, y, z
            if distance[0] > 0 and 0 in possible_actions:
                action = 0
            elif distance[0] < 0 and 3 in possible_actions:
                action = 3
            elif distance[1] > 0 and 1 in possible_actions:
                action = 1
            elif distance[1] < 0 and 4 in possible_actions:
                action = 4
            elif distance[2] > 0 and 2 in possible_actions:
                action = 2
            elif distance[2] < 0 and 5 in possible_actions:
                action = 5
            elif len(possible_actions) > 0:
                action = random.choice(possible_actions)
            else:
                action = 0 # chose a invalid action
            
            next_state, reward, done, _ = env_copy.step(action)  # Step with random action
            current_state = next_state
            total_reward += reward

        return total_reward

    def get_best_action_sequence(self, num_actions):
        # Generate the best sequence of actions by traversing the tree from the root
        current_node = self.tree.root
        action_sequence = []
        finish_2_pin_pair = False

        for _ in range(num_actions):
            if not current_node.done and len(current_node.children) > 0:
                best_child = self.tree.select_best_child(current_node)
                action_sequence.append(best_child.action)
            # if current_node.done and current_node.reward > 0:
            #     success_2_pin_pair = 1  
            # elif current_node.done and current_node.reward < 0:
            #     finish_2_pin_pair = -1
            finish_2_pin_pair = current_node.done
            current_node = best_child  # Move to the next best child
            if finish_2_pin_pair:
                if current_node.reward <= 0:
                    action_sequence = action_sequence[0:1] if len(action_sequence) > 1 else [0]
                break

        return action_sequence, finish_2_pin_pair
    


def solve_routing_problem(input_file_path):
    # Initialization
    state_size = 18  # Given state size
    action_size = 6  # 4 directions and 2 layer transitions

    grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name = load_input_file(input_file_path=input_file_path)

    env = RoutingEnv_v2(
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

    done = False  # Flag to indicate if the current episode is finished
    total_rewards = 0  # Accumulated rewards

    # Loop to solve the routing problem with episodes
    while True:
        # Initialize a MCTS and perform the action sequence
        mcts = MCTS(exploration_constant=0.7, num_simulations=50)
        action_sequence, finish_2_pin_pair = mcts.perform_mcts(env, env.state)
        node_visited = {}

        # Loop to execute the actions in the sequence
        for action in action_sequence:
            new_state, reward, done, _ = env.step(action)  # Step in the environment
            total_rewards += reward  # Accumulate rewards

        # Handle the 'done' flag
        if done:
            # If finished with one pair, move to the next
            if pin_pair_index < len(net_pin_pairs) - 1:
                if reward > 0:
                    pin_pair_index += 1  # Move to the next pair
                    env.update_env_info(Finish_pair=True)  # Update environment info
                env.init_new_pair_state(pin_pair_index)
                done = False
            elif pin_pair_index == len(net_pin_pairs) - 1:
                # Handling the last pair in the net
                if reward > 0:
                    # If completed this net, move to the next one
                    net_index += 1
                    if net_index < len(nets_mst):
                        # If more nets to solve
                        net_pin_pairs = nets_mst[net_index]
                        pin_pair_index = 0
                        env.update_env_info(Finish_pair=True, Finish_net=True)
                        env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)
                        done = False
                    else:
                        # If all nets are completed
                        env.update_env_info(Finish_pair=True, Finish_net=True)
                        print("Finish all nets")
                        print("Total rewards:", total_rewards)
                        print(env.nets_visited)
                        break
                else:
                    # If failed to finish the last pair, re-initialize state
                    env.init_new_pair_state(pin_pair_index)
                    done = False

    

    total_wirelength = 0
    for net_index in range (len(nets_mst)):
        print("net_index: ", net_index, env.nets_visited[net_index])
        total_wirelength += len(env.nets_visited[net_index]) - 1
    print("total wirelength: ", total_wirelength)

    mask_h = (env.capacity_info_h < 0) & (env.capacity_info_h > -10)
    mask_v = (env.capacity_info_v < 0) & (env.capacity_info_v > -10)
    overflow = np.sum(env.capacity_info_h[mask_h]) + np.sum(env.capacity_info_v[mask_v])
    print("overflow: ", overflow)

    total_congestion, min_capacity, total_wire_length = evaluation(env)

    # Return the total rewards or other relevant data if needed
    return total_congestion, min_capacity, total_wire_length


# Main block that calls the function
if __name__ == '__main__':
    file_path = 'benchmark/test_benchmark_6.gr'
    total_congestion, min_capacity, total_wire_length = solve_routing_problem(file_path)  # Call the function with the provided file path
    print("Total congestion:", total_congestion)
    print("Minimum capacity:", min_capacity)
    print("Total wire length:", total_wire_length)