import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple
from CustomRoutingEnv import CustomRoutingEnv as Env

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 3
        self.NUM_EPISODES = 30
        self.MAX_STEP = 40

class DQN_Agent:
    def __init__(self, env, capacity=100):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = 6 #env.action_space.shape
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(capacity)
        self.para = Parameter()
        self.steps_done = 0

    def select_action(self, state, output_dim, steps_done):
        sample = random.random()
        eps_threshold = self.para.EPS_END + (self.para.EPS_START - self.para.EPS_END) * \
            math.exp(-1. * self.steps_done / self.para.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(output_dim)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.para.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.para.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.para.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.para.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        
        for i_episode in range(self.para.NUM_EPISODES):
            i_reward_accum = 0
            self.env.reset()
            state = torch.tensor([self.env.current_state], dtype=torch.float32)
            for t in range(self.para.MAX_STEP):
                action = self.select_action(state, self.output_dim, t)
                next_state, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float32)

                if not done:
                    next_state = torch.tensor([next_state], dtype=torch.float32)
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()

                i_reward_accum += reward

                if done:
                    break
                
            if i_episode % self.para.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 5 == 0:
                print(f'EPOCH {i_episode} : ')
                print(state, reward, done, info)
                print("reward_accum: ", i_reward_accum)


        print('Training Complete')

    def test(self):
        total_reward = 0
        num_episodes = 10  # Number of episodes for testing

        for i_episode in range(num_episodes):
            self.env.reset()
            state = torch.tensor([self.env.current_state], dtype=torch.float32)  # Initialize the state tensor
            episode_reward = 0

            for t in range(self.para.MAX_STEP):
                action = self.select_action(state, self.output_dim, t)
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward

                state = torch.tensor([next_state], dtype=torch.float32) if not done else None

                if done:
                    break  # Exit the inner loop if the episode is done

            total_reward += episode_reward
            print(f'Episode {i_episode + 1}: Total Reward = {episode_reward}')

        average_reward = total_reward / num_episodes
        print(f'Average Reward over {num_episodes} episodes: {average_reward}')



if __name__ == '__main__':
    env = Env(input_file_path='benchmark_reduced/test_benchmark_1.gr')
    dqn = DQN_Agent(env)
    env.reset()
    dqn.train()
    dqn.test()