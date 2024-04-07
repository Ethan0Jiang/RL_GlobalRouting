import gym
from gym import spaces
import numpy as np
import MST as tree

class CustomRoutingEnv(gym.Env):
    def __init__(self, input_file_path):
        super(CustomRoutingEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # For example, move in XYZ directions and their negatives
        self.observation_space = spaces.Box(low=np.array([-np.inf]*12), high=np.array([np.inf]*12), dtype=np.float32)  # 12-dimensional state
        self.load_input_file(input_file_path)
        self.nets_mst = []
        pinList_allNet = self.prepareTwoPinList_allNet()
        for net_name, pin_pairs in pinList_allNet.items():
            self.nets_mst.append(tree.generateMST(pin_pairs))
        self.current_state = self.initialize_state()


    def initialize_state(self):
        # self.current_net_name = next(iter(self.nets_mst.keys()))
        self.net_index = 0
        self.pin_pair_index = 0
        first_mst = self.nets_mst[self.net_index][self.pin_pair_index]

        self.start_point = first_mst[0]
        self.current_point = self.start_point
        self.target_point = first_mst[1]

        state = np.zeros(12)
        state[0:3] = self.start_point
        state[3:6] = np.array(self.target_point) - np.array(self.start_point)

        # Calculate the capacity information for each grid cell
        self.capacity_info = np.zeros((self.grid_size[0], self.grid_size[1], self.grid_size[2]))
        for z in range(self.grid_size[2]):
            self.capacity_info[:, :, z] += self.horizontal_capacity[z]  # X direction
            self.capacity_info[:, :, z] += self.vertical_capacity[z]    # Y direction
        
        for adjustment in self.adjustments:
            x1, y1, z1, x2, y2, z2, change = adjustment[0] + adjustment[1] + [adjustment[2]]
            self.capacity_info[x1:x2+1, y1:y2+1, z1:z2+1] = change

        x, y, z = map(int, self.start_point)
        grid_x, grid_y, grid_z = self.grid_size

        # Set capacities for each direction, using -inf for out-of-bounds
        state[6] = self.capacity_info[x+1, y, z] if x+1 < grid_x else -np.inf
        state[7] = self.capacity_info[x, y+1, z] if y+1 < grid_y else -np.inf
        state[8] = self.capacity_info[x, y, z+1] if z+1 < grid_z else -np.inf
        state[9] = self.capacity_info[x-1, y, z] if x-1 >= 0 else -np.inf
        state[10] = self.capacity_info[x, y-1, z] if y-1 >= 0 else -np.inf
        state[11] = self.capacity_info[x, y, z-1] if z-1 >= 0 else -np.inf

        return state


    def scale_pins(self, pins): # input: 'pins':[[10, 12, 1], [42, 37, 1], [58, 17, 1], [77, 31, 1]]
        scaled_pins = []
        for pin in pins:
            # Scaling pin positions to tile numbers
            x_scaled = np.floor((pin[0] - self.grid_origin[0]) / self.grid_dimensions[0])
            y_scaled = np.floor((pin[1] - self.grid_origin[1]) / self.grid_dimensions[1])
            scaled_pins.append([x_scaled, y_scaled, pin[2]])  # Keeping the layer information unchanged
        return scaled_pins
        

    def load_input_file(self, input_file_path):
        # Initialize containers for the input data
        self.grid_size = None
        self.vertical_capacity = []
        self.horizontal_capacity = []
        self.minimum_width = []
        self.minimum_spacing = []
        self.via_spacing = []
        self.grid_origin = None
        self.grid_dimensions = None
        self.nets = {}
        self.nets_scaled = {}
        self.adjustments = []

        with open(input_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        # Parsing the fixed first 8 lines
        self.grid_size = tuple(map(int, lines[0].split()[1:]))
        self.vertical_capacity = list(map(int, lines[1].split()[2:]))
        self.horizontal_capacity = list(map(int, lines[2].split()[2:]))
        self.minimum_width = list(map(int, lines[3].split()[2:]))
        self.minimum_spacing = list(map(int, lines[4].split()[2:]))
        self.via_spacing = list(map(int, lines[5].split()[2:]))
        self.grid_origin = list(map(int, lines[6].split()[:2]))
        self.grid_dimensions = list(map(int, lines[6].split()[2:]))
        
        num_nets = int(lines[7].split()[2])
        current_line = 8

        # Parsing nets
        for _ in range(num_nets):
            net_header = lines[current_line].split()
            net_id = net_header[0]
            num_pins = int(net_header[2])
            self.nets[net_id] = {'net_info': net_header[1:], 'pins': []}
            self.nets_scaled[net_id] = {'net_info': net_header[1:], 'pins': []}
            current_line += 1
            
            # Parsing pins for the current net
            for _ in range(num_pins):
                pin_info = list(map(int, lines[current_line].split()))
                self.nets[net_id]['pins'].append(pin_info)
                pin_info_scaled = self.scale_pins([pin_info])[0]
                self.nets_scaled[net_id]['pins'].append(pin_info_scaled)
                current_line += 1

        # Parsing adjustments
        num_adjustments = int(lines[current_line])
        current_line += 1

        for _ in range(num_adjustments):
            adjustment_info = list(map(int, lines[current_line].split()))
            adjustment_info = [adjustment_info[0:3], adjustment_info[3:6], adjustment_info[6]]
            self.adjustments.append(adjustment_info)
            current_line += 1

    def prepareTwoPinList_allNet(self):
        twoPinList_allNet = {}
        
        # Iterate through each net in self.nets
        for net_name, net_info in self.nets_scaled.items():
            pins = net_info['pins']  # Extract the list of pins for the current net
            
            # Connect each pin to the next one to form a simple chain
            pin_pairs = []
            for i in range(len(pins)-1):  # Stop before the last pin to avoid index out of range
                # Create tuples for each pin representing (x, y, layer)
                pin1 = (pins[i][0], pins[i][1], pins[i][2])
                pin2 = (pins[i+1][0], pins[i+1][1], pins[i+1][2])
                pin_pairs.append([pin1, pin2])
            
            twoPinList_allNet[net_name] = pin_pairs
        
        return twoPinList_allNet




    def reset(self):
        self.current_state = self.initialize_state()

        return self.current_state
    

    def update_state(self):
        grid_x, grid_y, grid_z = self.grid_size

        # Update state information
        state_distance = np.array(self.target_point) - np.array(self.current_point)
        self.current_state[0:3] = self.current_point
        self.current_state[3:6] = state_distance

        # Update the edge capacities in the state vector
        move_mapping = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        for i, direction in enumerate(move_mapping):
            adjacent_location = np.array(self.current_point) + np.array(direction)
            if 0 <= adjacent_location[0] < grid_x and 0 <= adjacent_location[1] < grid_y and 0 <= adjacent_location[2] < grid_z:
                self.current_state[6 + i] = self.capacity_info[tuple(adjacent_location)]
            else:
                self.current_state[6 + i] = -np.inf
    

    def step(self, action):
        # Define move mapping for actions
        move_mapping = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        move = move_mapping[action]
        new_location = tuple(map(int, np.array(self.current_point) + np.array(move)))
        grid_x, grid_y, grid_z = self.grid_size

        # Initialize reward and done flag
        reward = -1  # Default reward for each step
        done = False

        # Check bounds and whether the new location exceeds capacity
        if 0 <= new_location[0] < grid_x and 0 <= new_location[1] < grid_y and 0 <= new_location[2] < grid_z and self.capacity_info[tuple(new_location)] > 0:
            # Valid move: update capacity at the current location
            self.capacity_info[int(self.current_point[0]), int(self.current_point[1]), int(self.current_point[2])] -= 1

            # Update the current point to the new location
            self.current_point = new_location

            # Update the state vector
            self.update_state()

            # Check if the target is reached
            if np.array_equal(self.current_point, self.target_point):
                reward = 10  # Reward for reaching the target
                # Move to the next start and target pin
                self.pin_pair_index += 1
                if self.pin_pair_index >= len(self.nets_mst[self.net_index]):
                    # All pin pairs in the current net processed; move to the next net
                    self.net_index += 1
                    self.pin_pair_index = 0
                    if self.net_index >= len(self.nets_mst):
                        # All nets processed; episode done
                        done = True
                        reward = 1000
                    else:
                        # Reset to the first pin pair of the next net
                        next_mst = self.nets_mst[self.net_index][self.pin_pair_index]
                        self.start_point, self.target_point = next_mst
                        self.current_point = self.start_point
                        self.update_state()


                else:
                    # Move to the next pin pair in the current net
                    next_mst = self.nets_mst[self.net_index][self.pin_pair_index]
                    self.start_point, self.target_point = next_mst
                    self.current_point = self.start_point
                    self.update_state()

        # if the new location is out of bounds or exceeds capacity, do not update the current point or capacity      
        else:
            # Invalid move or exceeded capacity: do not update the current point or capacity
            pass  # The reward remains -1 as initialized

        return self.current_state, reward, done, {}




if __name__ == '__main__':
    # Example usage
    env = CustomRoutingEnv(input_file_path='benchmark_reduced/test_benchmark_1.gr')
    env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(next_state, reward, done, info)
