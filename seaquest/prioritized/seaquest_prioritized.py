#!/usr/bin/env python3
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
import random
random.seed(5)
import numpy as np
np.random.seed(5)
import shutil

import gym
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

# from lib import dqn_model, common
# from other import actions, agent, experience
from wrappers import wrap_dqn
import csv
import torch.nn as nn
import collections
import copy
from collections import namedtuple
from torch.autograd import Variable


import os; os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from priority_queue import Memory
import sys
sys.path.insert(0, '..')
from prioritized_buffer import PrioritizedReplayBuffer as Memory

# Some scores for comparison right here:
# https://github.com/chainer/chainerrl/tree/master/examples/atari/dqn

# Random: 12850
# Human score: 29028.0
# DQN Score 85641.0

# use this to implement prioritization: https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb


HYPERPARAMS = {
        'replay_size':      10000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
}



class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
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


    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)


class Dueling_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )


        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=7 * 7 * 64, out_features=1)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.num_actions)

        return val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


class RewardTracker:
    def __init__(self, length=100, stop_reward=20):
        self.stop_reward = stop_reward
        self.length = length
        self.rewards = []
        self.position = 0
        self.stop_reward = stop_reward
        self.mean_score = 0

    def add(self, reward):
        if len(self.rewards) < self.length:
            self.rewards.append(reward)
        else:
            self.rewards[self.position] = reward
            self.position = (self.position + 1) % self.length
        self.mean_score = np.mean(self.rewards)

    def meanScore(self):
        return self.mean_score


class EpsilonTracker:
    def __init__(self, params):
        self._epsilon = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_delta = 1.0 * (params['epsilon_start'] - params['epsilon_final']) / params['epsilon_frames']

    def epsilon(self):
        old_epsilon = self._epsilon
        self._epsilon -= self.epsilon_delta
        return max(old_epsilon, self.epsilon_final)
    @property
    def currentEpsilon(self):
        return max(self._epsilon, self.epsilon_final)


# class ReplayMemory(object):
#     def __init__(self, capacity, transition):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#         self.transition = transition
#
#     def push(self, *args):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = self.transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


class Trainer(object):
    def __init__(self):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make('Seaquest-v0')
        self.env = wrap_dqn(self.env)

        self.policy_net = DQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_tracker = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        # self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.memory = Memory(self.params['replay_size'], self.transition)
        self.episode = 0
        self.state = self.preprocess(self.env.reset())
        self.score = 0
        self.batch_size = self.params['batch_size']
        self.tb_writer = SummaryWriter('results')


    def preprocess(self, state):
        state = torch.tensor(np.expand_dims(state, 0)).to(self.device)
        return state.float() / 256


    def addExperience(self):
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.env.action_space.n)], device=self.device)
        else:
            action = torch.argmax(self.policy_net(self.state), dim=1).to(self.device)
        next_state, reward, done, _ = self.env.step(action.item())
        next_state = self.preprocess(next_state)
        self.score += reward
        self.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state, done)
        if done:
            self.state = self.preprocess(self.env.reset())
            self.episode += 1
        else:
            self.state = next_state
        return done


    def optimizeModel(self):
        transitions, ISWeights, tree_idx = self.memory.sample(self.batch_size)
        ISWeights = torch.tensor(ISWeights, device=self.device)
        batch = self.transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
        abs_errors = abs(expected_state_action_values.unsqueeze(1) - state_action_values)
        loss = torch.sum(abs_errors ** 2 * ISWeights) / self.batch_size
        abs_errors_clone = abs_errors.clone().detach().cpu().numpy()
        try:
            self.memory.update_priorities(tree_idx, abs_errors_clone + 1e-3)
        except:
            import pdb; pdb.set_trace()
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        frame_idx = 0
        for episode in range(int(1e7)):
            frame_idx += 1
            # play one move
            game_over = self.addExperience()

            # is this round over?
            if game_over:
                self.reward_tracker.add(self.score)
                self.tb_writer.add_scalar('Mean Score', self.reward_tracker.meanScore(), episode)
                self.tb_writer.add_scalar('Score', self.score, episode)
                self.tb_writer.add_scalar('Epsilon', self.epsilon_tracker.currentEpsilon, episode)
                print('Game: %s Score: %s Mean Score: %s' % (self.episode, self.score, self.reward_tracker.meanScore()))
                self.score = 0

            # are we done prefetching?
            if not self.memory.donePrefetching():
                continue
            self.optimizeModel()
            if frame_idx % self.params['target_net_sync'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        torch.save(self.policy_net, 'final.pth')


    def playback(self, path):
        target_net = torch.load(path, map_location='cpu')
        env = gym.make('PongNoFrameskip-v4')
        env = wrap_dqn(env)
        state = self.preprocess(env.reset())
        done = False
        score = 0
        import time
        while not done:
            time.sleep(0.015)
            action = torch.argmax(target_net(state), dim=1).to(self.device)
            state, reward, done, _ = env.step(action.item())
            state = self.preprocess(state)
            score += reward
        print("Score: ", score)

def cleanup():
    if os.path.isdir('results'):
        shutil.rmtree('results')
    csv_txt_files = [x for x in os.listdir('.') if '.TXT' in x or '.csv' in x]
    for csv_txt_file in csv_txt_files:
        os.remove(csv_txt_file)


if __name__ == "__main__":
    cleanup()
    print('Creating Trainer Object')
    trainer = Trainer()
    print('Trainer Initialized')
    trainer.train()






