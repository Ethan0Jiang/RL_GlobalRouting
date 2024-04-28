import gym
from gym import spaces
import numpy as np
import MST as tree

class RoutingEnv(gym.Env):
    def __init__(self, grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, adjustments):
        super(RoutingEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # For example, move in XYZ directions and their negatives
        self.observation_space = spaces.Box(low=np.array([-np.inf]*18), high=np.array([np.inf]*18), dtype=np.float32)  # 18-dimensional state
        self.grid_size = grid_size
        self.vertical_capacity = vertical_capacity
        self.horizontal_capacity = horizontal_capacity
        self.minimum_width = minimum_width
        self.minimum_spacing = minimum_spacing
        self.via_spacing = via_spacing
        self.grid_origin = grid_origin
        self.grid_dimensions = grid_dimensions
        self.adjustments = adjustments

        self.net_pin_pairs = [] # a list of 2 pin pair nets

        self.nets_visited = {}  # Dictionary to store visited locations for each net
        self.net_index = None
        self.pin_pair_index = None
        self.start_point = None
        self.current_point = None  
        self.target_point = None

        self.capacity_info_h = None
        self.capacity_info_v = None
        self.state = None

        grid_x, grid_y, grid_z = self.grid_size
        capacity_info_h_size = (grid_x-1, grid_y, grid_z)  # Adjusted order to (x, y, z)
        capacity_info_v_size = (grid_x, grid_y-1, grid_z)  # Adjusted order to (x, y, z)


        # Calculate the capacity information for each grid cell
        self.capacity_info_h = np.zeros(capacity_info_h_size) # horizontal capacity, specify on the right side of the cell, of the edge
        self.capacity_info_v = np.zeros(capacity_info_v_size) # vertical capacity, specify on the top side of the cell, of the edge
    

        for z in range(grid_z):
            self.capacity_info_h[:, :,z] = self.horizontal_capacity[z] if self.horizontal_capacity[z] != 0 else -np.inf # horizontal capacity
            self.capacity_info_v[:, :,z] = self.vertical_capacity[z] if self.vertical_capacity[z] != 0 else -np.inf # vertical capacity
        
        for adjustment in self.adjustments:
            x1, y1, z1, x2, y2, z2, change = adjustment[0] + adjustment[1] + [adjustment[2]]
            if change == 0:
                change = -np.inf # set the capacity to -inf if the capacity is 0
            # fix the ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            if z1 > z2:
                z1, z2 = z2, z1
    
            if x1 != x2 and y1 == y2 and z1 == z2: # horizontal change
                self.capacity_info_h[x1:x2, y1, z1] = change
            elif x1 == x2 and y1 != y2 and z1 == z2: # vertical change
                self.capacity_info_v[x1, y1:y2, z1] = change
            else:
                raise ValueError('Invalid adjustment: ', adjustment, 'Only horizontal or vertical adjustments are supported')
            

    def update_env_info(self, Finish_pair=False, Finish_net=False):
        if Finish_pair:
            # Update the visited locations for the current net, do this before init_new_pair function
            for location in self.pin_pair_visited[self.pin_pair_index]:
                self.nets_visited.setdefault(self.net_index, set()).add(location)
            # Update the capacity information in init_new_pair function
            self.capacity_info_h = self.capacity_info_h_temp
            self.capacity_info_v = self.capacity_info_v_temp

    def init_new_net_state(self, net_index, pin_pair_index, net_pin_pairs):
        self.fail_count = 0
        self.pin_pair_idxs = set()
        self.pin_pair_idxs.add(pin_pair_index)
        self.net_index = net_index
        self.pin_pair_index = pin_pair_index
        self.net_pin_pairs = net_pin_pairs
        first_pair = self.net_pin_pairs[self.pin_pair_index]

        self.start_point = tuple(map(int, first_pair[0]))
        self.current_point = self.start_point
        self.target_point = tuple(map(int, first_pair[1]))

        self.capacity_info_h_temp = self.capacity_info_h.copy()
        self.capacity_info_v_temp = self.capacity_info_v.copy()

        self.pin_pair_visited = [set() for _ in range(len(self.net_pin_pairs))]

        return self.update_state()
    
    def init_new_pair_state(self, pin_pair_index):
        self.fail_count = 0
        if self.pin_pair_index == pin_pair_index:
            print(" repeat the same pin pair index:", pin_pair_index, "for the net index:", self.net_index)
        elif pin_pair_index in self.pin_pair_idxs:
            raise ValueError('The pin pair index is already visited')
        self.pin_pair_idxs.add(pin_pair_index)

        self.pin_pair_index = pin_pair_index
        pair = self.net_pin_pairs[self.pin_pair_index]
        self.pin_pair_visited[self.pin_pair_index] = set()

        self.start_point = tuple(map(int, pair[0]))
        self.current_point = self.start_point
        self.target_point = tuple(map(int, pair[1]))

        self.capacity_info_h_temp = self.capacity_info_h.copy()
        self.capacity_info_v_temp = self.capacity_info_v.copy()

        return self.update_state_v2()
    
    def safe_access(self, arr, indices):
        # Check if all indices are within the valid range
        if all(0 <= idx < dim for idx, dim in zip(indices, arr.shape)):
            return arr[tuple(indices)]
        else:
            return -np.inf
    

    def check_valid_move(self, action, new_location):
        
        if new_location in self.pin_pair_visited[self.pin_pair_index]: # if the new location is visited by the current 2-pin pair
            return False

        if action == 0 and self.state[6] > -np.inf: # +x
            return True
        elif action == 1 and self.state[7] > -np.inf: # +y
            return True
        elif action == 3 and self.state[8] > -np.inf: # -x
            return True
        elif action == 4 and self.state[9] > -np.inf: # -y
            return True
        elif action == 2 and (self.state[10] > -np.inf or self.state[11] > -np.inf 
                            or self.state[12] > -np.inf or self.state[13] > -np.inf): # +z
            return True
        elif action == 5 and (self.state[14] > -np.inf or self.state[15] > -np.inf 
                            or self.state[16] > -np.inf or self.state[17] > -np.inf): # -z
            return True
        else:
            return False
    
    def get_possible_actions(self):
        # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z
        # move_mapping = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        actions_TF = [False, False, False, False, False, False]
        action = []
        if self.state[6] > -np.inf and self.current_point+(1,0,0) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[0] = True
            action.append(0)
        if self.state[7] > -np.inf and self.current_point+(0,1,0) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[1] = True
            action.append(1)
        if self.state[8] > -np.inf and self.current_point+(-1,0,0) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[3] = True
            action.append(3)
        if self.state[9] > -np.inf and self.current_point+(0,-1,0) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[4] = True
            action.append(4)
        if self.state[10] > -np.inf or self.state[11] > -np.inf or self.state[12] > -np.inf or self.state[13] > -np.inf and self.current_point+(0,0,1) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[2] = True
            action.append(2)
        if self.state[14] > -np.inf or self.state[15] > -np.inf or self.state[16] > -np.inf or self.state[17] > -np.inf and self.current_point+(0,0,-1) not in self.pin_pair_visited[self.pin_pair_index]:
            actions_TF[5] = True
            action.append(5)
        return actions_TF, action
        
    def update_state(self):
        state = np.zeros(18)
        # Update state information
        state_distance = np.array(self.target_point) - np.array(self.current_point)
        state[0:3] = self.current_point
        state[3:6] = state_distance

        x, y, z = self.current_point  # Correct the order of coordinates for proper indexing
        
        # Set capacities for each direction, using -inf for out-of-bounds, or invalid moves
        # Update edge capacities with correct indexing
        state[6] = self.safe_access(self.capacity_info_h, (x, y, z))  # x direction:+1
        state[7] = self.safe_access(self.capacity_info_v, (x, y, z))  # y direction:+1
        state[8] = self.safe_access(self.capacity_info_h, (x-1, y, z))  # x direction:-1
        state[9] = self.safe_access(self.capacity_info_v, (x, y-1, z))  # y direction:-1

        # above, z+1
        state[10] = self.safe_access(self.capacity_info_h, (x, y, z+1))  # x direction:+1
        state[11] = self.safe_access(self.capacity_info_v, (x, y, z+1))  # y direction:+1
        state[12] = self.safe_access(self.capacity_info_h, (x-1, y, z+1))  # x direction:-1
        state[13] = self.safe_access(self.capacity_info_v, (x, y-1, z+1))  # y direction:-1

        # below, z-1
        state[14] = self.safe_access(self.capacity_info_h, (x, y, z-1))  # x direction:+1 in z-1 level
        state[15] = self.safe_access(self.capacity_info_v, (x, y, z-1))  # y direction:+1 in z-1 level
        state[16] = self.safe_access(self.capacity_info_h, (x-1, y, z-1))  # x direction:-1 in z-1 level
        state[17] = self.safe_access(self.capacity_info_v, (x, y-1, z-1))  # y direction:-1 in z-1 level

        self.state = state
        return state

    
    def update_state_v2(self):
        state = np.zeros(18)
        # Update state information
        state_distance = np.array(self.target_point) - np.array(self.current_point)
        state[0:3] = self.current_point
        state[3:6] = state_distance

        x, y, z = self.current_point  # Correct the order of coordinates for proper indexing
        
        # Set capacities for each direction, using -inf for out-of-bounds, or invalid moves
        # Update edge capacities with correct indexing
        state[6] = self.safe_access(self.capacity_info_h, (x, y, z)) if (x+1, y, z) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf  # x direction:+1
        state[7] = self.safe_access(self.capacity_info_v, (x, y, z)) if (x, y+1, z) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf  # y direction:+1
        state[8] = self.safe_access(self.capacity_info_h, (x-1, y, z))  if (x-1, y, z) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # x direction:-1
        state[9] = self.safe_access(self.capacity_info_v, (x, y-1, z))  if (x, y-1, z) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # y direction:-1

        # above, z+1
        state[10] = self.safe_access(self.capacity_info_h, (x, y, z+1))  if (x+1, y, z+1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # x direction:+1
        state[11] = self.safe_access(self.capacity_info_v, (x, y, z+1))  if (x, y+1, z+1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # y direction:+1
        state[12] = self.safe_access(self.capacity_info_h, (x-1, y, z+1))  if (x-1, y, z+1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # x direction:-1
        state[13] = self.safe_access(self.capacity_info_v, (x, y-1, z+1))  if (x, y-1, z+1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf # y direction:-1

        # below, z-1
        state[14] = self.safe_access(self.capacity_info_h, (x, y, z-1))  if (x+1, y, z-1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf
        state[15] = self.safe_access(self.capacity_info_v, (x, y, z-1))  if (x, y+1, z-1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf
        state[16] = self.safe_access(self.capacity_info_h, (x-1, y, z-1))  if (x-1, y, z-1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf
        state[17] = self.safe_access(self.capacity_info_v, (x, y-1, z-1))  if (x, y-1, z-1) not in self.pin_pair_visited[self.pin_pair_index] else -np.inf

        self.state = state
        return state
        
    

    def step(self, action):
        # Define move mapping for actions
        # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z
        move_mapping = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        move = move_mapping[action]
        new_location = tuple(map(int, np.array(self.current_point) + np.array(move)))

        # Initialize reward and done flag
        reward = -1  # Default reward for each step
        done = False
        # # Penalize for moving away from the target
        # if np.dot(np.array(move), np.array(self.target_point) - np.array(self.current_point)) < 0:
        #     reward -= 1
        print("current_point: ", self.current_point)
        print("fail_count: ", self.fail_count)

        # check if the new location is valid
        if self.check_valid_move(action, new_location):
            self.fail_count = 0
            # Update the capacity information
            if new_location in self.nets_visited.get(self.net_index, set()):
                reward = 0 # non-Penalize for revisiting a location of the privous 2pin pair
            elif action == 0:
                self.capacity_info_h_temp[self.current_point[0], self.current_point[1], self.current_point[2]] -= 1
            elif action == 1:
                self.capacity_info_v_temp[self.current_point[0], self.current_point[1], self.current_point[2]] -= 1
            elif action == 3:
                self.capacity_info_h_temp[self.current_point[0]-1, self.current_point[1], self.current_point[2]] -= 1
            elif action == 4:
                self.capacity_info_v_temp[self.current_point[0], self.current_point[1]-1, self.current_point[2]] -= 1

            # Update the visited locations for the current pin pair
            self.pin_pair_visited[self.pin_pair_index].add(self.current_point)
            self.current_point = new_location
            self.update_state_v2()

            if action == 2 or action == 5:
                reward = -1 # discourage moving in z direction, avoid vias # maybe not needed, since in nature move in z will follow by move in x or y, 2*-1 panenty
            
            if new_location == self.target_point:
                print("Finish a 2-pin pair", self.current_point, self.target_point)
                reward = 1000
                done = True
            elif self.get_possible_actions() == [False, False, False, False, False, False]:
                print("No possible further move")
                done = True
                reward = -10 * np.linalg.norm(np.array(self.target_point) - np.array(self.current_point))

        else:
            reward = -5
            self.fail_count += 1
            if self.fail_count > 5:
                done = True
                # the closer to the target, the higher the reward
                reward = -10 * np.linalg.norm(np.array(self.target_point) - np.array(self.current_point))
                print("Fail to move to the next location, exceed the fail count")

        return self.state, reward, done, {}





def load_input_file(input_file_path):
        # Initialize containers for the input data
        grid_size = None
        vertical_capacity = []
        horizontal_capacity = []
        minimum_width = []
        minimum_spacing = []
        via_spacing = []
        grid_origin = None
        grid_dimensions = None
        nets = {}
        nets_scaled = {}
        adjustments = []
        net_name2id = {}
        net_id2name = []

        # Helper function to scale pin positions to grid coordinates
        def scale_pins(pins, grid_origin, grid_dimensions):
            scaled_pins = []
            for pin in pins:
                # Scaling pin positions to tile numbers
                x_scaled = np.floor((pin[0] - grid_origin[0]) / grid_dimensions[0])
                y_scaled = np.floor((pin[1] - grid_origin[1]) / grid_dimensions[1])
                scaled_pins.append([x_scaled, y_scaled, pin[2]])  # Keeping the layer information unchanged
            return scaled_pins
        
        with open(input_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        # Parsing the fixed first 8 lines
        grid_size = tuple(map(int, lines[0].split()[1:]))
        vertical_capacity = list(map(int, lines[1].split()[2:]))
        horizontal_capacity = list(map(int, lines[2].split()[2:]))
        minimum_width = list(map(int, lines[3].split()[2:]))
        minimum_spacing = list(map(int, lines[4].split()[2:]))
        via_spacing = list(map(int, lines[5].split()[2:]))
        grid_origin = list(map(int, lines[6].split()[:2]))
        grid_dimensions = list(map(int, lines[6].split()[2:]))

        num_nets = int(lines[7].split()[2])
        current_line = 8

        # Parsing nets
        for _ in range(num_nets):
            net_header = lines[current_line].split()
            net_name = net_header[0]
            net_id = int(net_header[1])
            net_name2id[net_name] = net_id
            net_id2name.append(net_name)
            num_pins = int(net_header[2])
            nets[net_id] = {'net_info': net_header[:], 'pins': []}
            nets_scaled[net_id] = {'net_info': net_header[:], 'pins': []}
            current_line += 1

            # Parsing pins for the current net
            for _ in range(num_pins):
                pin_info = list(map(int, lines[current_line].split()))
                pin_info[2] -= 1  # Adjusting the layer index to start from 0
                nets[net_id]['pins'].append(pin_info)
                pin_info_scaled = scale_pins([pin_info], grid_origin, grid_dimensions)[0]
                nets_scaled[net_id]['pins'].append(pin_info_scaled)
                current_line += 1

        # Parsing adjustments
        num_adjustments = int(lines[current_line])
        current_line += 1

        for _ in range(num_adjustments):
            adjustment_info = list(map(int, lines[current_line].split()))
            adjustment_info[2] -= 1  # Adjusting the layer index to start from 0
            adjustment_info[5] -= 1  # Adjusting the layer index to start from 0
            adjustment_info = [adjustment_info[0:3], adjustment_info[3:6], adjustment_info[6]]
            adjustments.append(adjustment_info)
            current_line += 1

        return grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name

def prepareTwoPinList_allNet(nets_scaled):
    twoPinList_allNet = {}
    
    # Iterate through each net in nets_scaled
    for net_name, net_info in nets_scaled.items():
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

if __name__ == '__main__':

    grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name = load_input_file(input_file_path='benchmark_reduced/test_benchmark_1.gr')

    env = RoutingEnv(grid_size=grid_size, vertical_capacity=vertical_capacity, horizontal_capacity=horizontal_capacity, 
                     minimum_width=minimum_width, minimum_spacing=minimum_spacing, via_spacing=via_spacing, 
                     grid_origin=grid_origin, grid_dimensions=grid_dimensions, adjustments=adjustments)
    
    nets_mst = []
    pinList_allNet = prepareTwoPinList_allNet(nets_scaled)
    for net_id, pin_pairs in pinList_allNet.items():
        nets_mst.append(tree.generateMST(pin_pairs))

    net_index = 0
    net_pin_pairs = nets_mst[net_index]
    pin_pair_index = 0
    env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)
    done = False
    episodes_per_pair = 10  # Repeat the routing of each 2-pin pair for 10 episodes
    episodes_seccess = 0
    while_loop_count = 0
    while not done:
        while_loop_count += 1
        action = env.action_space.sample()
        # print("action: ", action)
        next_state, reward, done, info = env.step(action)
        # print(next_state, reward, done, info) # done is finish a 2-pin pair
        if done:
            if pin_pair_index < len(net_pin_pairs) - 1:
                if reward > 0:
                    if episodes_seccess < episodes_per_pair:
                        episodes_seccess += 1
                        env.init_new_pair_state(pin_pair_index) # update the self.nets_visited, update the self.capacity_info
                    else: # finish the episodes for the current 2-pin pair
                        pin_pair_index = pin_pair_index + 1 # move to the next 2-pin pair
                        env.update_env_info(Finish_pair=True)
                        env.init_new_pair_state(pin_pair_index)
                        episodes_seccess = 0
                    done = False
                else: # fail to move to the next location
                    env.init_new_pair_state(pin_pair_index)
                    done = False
            elif pin_pair_index == len(net_pin_pairs) - 1 and reward <=0: # fail to finish the last 2-pin pair
                env.init_new_pair_state(pin_pair_index)
                done = False
            elif pin_pair_index == len(net_pin_pairs) - 1 and reward > 0:
                if episodes_seccess < episodes_per_pair:
                    episodes_seccess += 1
                    env.init_new_pair_state(pin_pair_index)
                    done = False
                else:
                    net_index = net_index + 1
                    episodes_seccess = 0
                    if net_index < len(nets_mst):
                        net_pin_pairs = nets_mst[net_index]
                        pin_pair_index = 0
                        env.update_env_info(Finish_pair=True, Finish_net=True)
                        env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)
                        done = False
                    else:
                        # all nets are finished
                        env.update_env_info(Finish_pair=True, Finish_net=True)
                        break
    print("Finish all nets ", while_loop_count)
