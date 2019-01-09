import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple
import torch.optim as optim

np.random.seed(5)
random.seed(5)
torch.backends.cudnn.deterministic = True
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)



HYPERPARAMS = {
        'replay_size':      10000,
        'replay_initial':   9000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'episodes':         1e6
}

TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = TRANSITION

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EpsilonTracker:
    def __init__(self):
        self._epsilon = HYPERPARAMS['epsilon_start']
        self.epsilon_final = HYPERPARAMS['epsilon_final']
        self.epsilon_delta = 1.0 * (HYPERPARAMS['epsilon_start'] - HYPERPARAMS['epsilon_final']) / HYPERPARAMS['epsilon_frames']

    def epsilon(self):
        old_epsilon = self._epsilon
        self._epsilon -= self.epsilon_delta
        return max(old_epsilon, self.epsilon_final)



class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, device):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out = self.conv(Variable(torch.zeros(1, *input_shape)))
        conv_out_size = int(np.prod(conv_out.size()))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.num_actions = num_actions
        self.counter = 0
        self.device = device
        self.memory = ReplayMemory(HYPERPARAMS['replay_size'])
        self.epsilon_tracker = EpsilonTracker()
        self.optimizer = optim.Adam(self.parameters(), lr=HYPERPARAMS['learning_rate'])


    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)


    def getAction(self, state):
        if random.random() < self.epsilon_tracker.epsilon():
            return torch.tensor([random.randrange(self.num_actions)], device=self.device)
        else:
            return torch.argmax(self(state), dim=1).to(self.device)

    def optimizeModel(self, target_net):
        self.counter += 1
        transitions = self.memory.sample(HYPERPARAMS['batch_size'])
        batch = TRANSITION(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        state_action_values = self(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(HYPERPARAMS['batch_size'], device=self.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * HYPERPARAMS['gamma']) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.counter % HYPERPARAMS['target_net_sync'] == 0:
            target_net.load_state_dict(self.state_dict())



class Dueling_DQN(DQN):
    def __init__(self, input_shape, num_actions, device):
        super(Dueling_DQN, self).__init__(input_shape, num_actions, device)
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out = self.conv(Variable(torch.zeros(1, *input_shape)))
        conv_out_size = int(np.prod(conv_out.size()))
        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.num_actions = num_actions
        self.counter = 0
        self.device = device
        self.memory = ReplayMemory(HYPERPARAMS['replay_size'])
        self.epsilon_tracker = EpsilonTracker()
        self.optimizer = optim.Adam(self.parameters(), lr=HYPERPARAMS['learning_rate'])

    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.num_actions)
        return val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


