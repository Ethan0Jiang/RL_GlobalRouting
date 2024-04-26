import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import random
import math
from collections import namedtuple
import MST as tree
# from CustomRoutingEnv import CustomRoutingEnv as Env
from RoutingEnv import RoutingEnv as Env
from RoutingEnv import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128, track_running_stats=False),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(64, 32),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        
        x = self.layers(x)
        
        return x
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Parameter:
    def __init__(self):
        self.BATCH_SIZE = 3
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.LEARNING_RATE = 0.00001
        self.TARGET_UPDATE = 3
        self.NUM_EPISODES = 50
        self.MAX_STEP = 30

class DQN_Agent:
    def __init__(self, env, capacity=100):
        self.para = Parameter()
        self.env = env
        self.input_dim = 18 #env.observation_space.shape[0]
        self.output_dim = 6 #env.action_space.shape
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.para.LEARNING_RATE)
        self.clip_value = 1.0 
        self.memory = ReplayMemory(capacity)
        self.steps_done = 0
        self.total_step_done = 0
        self.mask = [1] * self.output_dim # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z

    def select_action(self, state):
        # state = torch.tensor(state, dtype=torch.float32)
        sample = random.random()
        eps_threshold = self.para.EPS_END + (self.para.EPS_START - self.para.EPS_END) * \
            math.exp(-1. * self.steps_done / self.para.EPS_DECAY)
        # self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
        
    def update_mask(self):
        # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z
        move_mapping = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        for i in range (self.output_dim):
            move = move_mapping[i]
            new_location = tuple(map(int, np.array(env.current_point) + np.array(move)))
            if not env.check_valid_move(i, new_location):
                self.mask[i] = -np.inf
            else:
                self.mask[i] = 1
    
    def select_action_with_mask(self, state, exploit=False):
        # state = torch.tensor(state, dtype=torch.float32)
        self.update_mask()
        sample = random.random()
        # eps_threshold = self.para.EPS_END + (self.para.EPS_START - self.para.EPS_END) * \
        #     math.exp(-1. * self.steps_done / self.para.EPS_DECAY)
        eps_threshold = 0.1
        # self.steps_done += 1
        if exploit:
            eps_threshold = 0

        mask = np.isneginf(state)
        state[mask] = -10
        # assert (state >=-50).all()
        
        # print("q_values: ",self.policy_net(state))
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = torch.tensor(np.multiply(np.array(self.policy_net(state)), np.array(self.mask)))
                # q_values = q_values.unsqueeze(0)
                return q_values.max(1)[1].view(1, 1)
        else:
            actions = [i for i in range (self.output_dim) if self.mask[i] != -np.inf]
            if len(actions) == 0:
                print("No valid action!!!")
                return torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
            return torch.tensor([[random.choice(actions)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.para.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.para.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Convert NumPy arrays to PyTorch tensors
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.para.BATCH_SIZE)
        mask = np.isneginf(non_final_next_states)
        non_final_next_states[mask] = -10
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.para.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='mean', beta=1.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()


    def train(self): 
        print("///// Start Training /////////")

        for i_episode in range(self.para.NUM_EPISODES):
            
            i_reward_accum = 0  
            state = torch.tensor([self.env.state], dtype=torch.float32)
            # print("state: ", state)

            for t in range(self.para.MAX_STEP):
                # print("total_step_done: ",self.total_step_done)
              
                action = self.select_action_with_mask(state) 
                next_state, reward, done, info = env.step(action.item())
                i_reward_accum += reward
                reward = torch.tensor([reward], dtype=torch.float32)
                self.memory.push(state, action, next_state, reward)
                

                self.optimize_model()

                next_state = torch.tensor([next_state], dtype=torch.float32)
                state = next_state

                self.total_step_done += 1

                if done:
                    if reward > 0:
                        print(f'EPOCH {i_episode} STEP {t} SUCCESS!!!!!!!')
                        break  # Exit the inner loop if the episode is done
                    else:
                        env.init_new_pair_state(pin_pair_index)
                        done = False
                        
            if i_episode % self.para.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 5 == 0:
                print(f'EPOCH {i_episode} : ')
                # print(state, reward, done, info)
                print("reward_accum: ", i_reward_accum)

        return done

    def test(self):
        print("///// Start Testing /////////")
        total_reward = 0
        num_episodes = 5  # Number of episodes for testing

        for i_episode in range(num_episodes):
            # self.env.reset()
            state = torch.tensor([self.env.state], dtype=torch.float32)  # Initialize the state tensor
            episode_reward = 0

            for t in range(self.para.MAX_STEP):
                self.steps_done = t
                action = self.select_action_with_mask(state)
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward

                if done:
                    if reward > 0:
                        print(f'EPOCH {i_episode} STEP {t} SUCCESS!!!!!!!')
                        break  # Exit the inner loop if the episode is done
                    else:
                        env.init_new_pair_state(pin_pair_index)
                        done = False

            state = torch.tensor([next_state], dtype=torch.float32) if not done else None
            total_reward += episode_reward
            print(f'Episode {i_episode + 1}: Total Reward = {episode_reward}')

        average_reward = total_reward / num_episodes
        print(f'Average Reward over {num_episodes} episodes: {average_reward}')
        print('///////Testing Complete////////')
  


if __name__ == '__main__':
    # Load input file and prepare environment
    grid_size, vertical_capacity, horizontal_capacity, minimum_width, minimum_spacing, via_spacing, grid_origin, grid_dimensions, nets, nets_scaled, adjustments, net_name2id, net_id2name = load_input_file(input_file_path='benchmark_reduced/test_benchmark_1.gr')
    env = Env(grid_size=grid_size, vertical_capacity=vertical_capacity, horizontal_capacity=horizontal_capacity, 
              minimum_width=minimum_width, minimum_spacing=minimum_spacing, via_spacing=via_spacing, 
              grid_origin=grid_origin, grid_dimensions=grid_dimensions, adjustments=adjustments)
    
    agent = DQN_Agent(env)

    # Generate minimum spanning trees for nets
    nets_mst = []
    pinList_allNet = prepareTwoPinList_allNet(nets_scaled)
    for net_id, pin_pairs in pinList_allNet.items():
        nets_mst.append(tree.generateMST(pin_pairs))
    print("nets_mst: ", nets_mst)

    # Start routing process
    failed_pin_pairs = [] # [net_index, pin_pair_index]
    for net_index in range (len(nets_mst)):
        
        net_pin_pairs = nets_mst[net_index]
        for pin_pair_index in range (len(net_pin_pairs)):
            if pin_pair_index == 0:
                env.init_new_net_state(net_index, pin_pair_index, net_pin_pairs)

            done = agent.train()
            if not done:
                failed_pin_pairs.append([net_index, pin_pair_index])
            env.init_new_pair_state(pin_pair_index)
            env.update_env_info(Finish_pair=True, Finish_net=(pin_pair_index==len(net_pin_pairs)-1))


    print("failed_pin_pairs: ", failed_pin_pairs)
    print("total_step_done: ", agent.total_step_done)
    print("Finish all nets")

    # agent.test()




    
