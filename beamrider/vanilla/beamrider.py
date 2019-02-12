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

from wrappers import wrap_dqn
import torch.nn as nn
import collections
import copy
from collections import namedtuple
from torch.autograd import Variable
import os


import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
from queues import PrioritizedReplayBuffer as Memory


HYPERPARAMS = {
        'replay_size':      50000,
        'replay_initial':   1000,
        'target_net_sync':  500,
        'total_frames':     10**5,
        'epsilon_frames':   10**4,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    5e-4,
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
        self.epsilon_delta = 1.0 * (params['epsilon_start'] - params['epsilon_final']) / params['total_frames']*0.1

    def epsilon(self):
        old_epsilon = self._epsilon
        self._epsilon -= self.epsilon_delta
        return max(old_epsilon, self.epsilon_final)
    @property
    def currentEpsilon(self):
        return max(self._epsilon, self.epsilon_final)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = tuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LinearScheduler(object):
    def __init__(self, start, stop, delta=None, timespan=None):
        assert delta or timespan
        self.value = start
        self.stop = stop
        if delta:
            self.delta = float(delta)
        elif timespan:
            self.delta = (stop - start) / float(timespan)

    def updateAndGetValue(self):
        self.value += self.delta
        return self.observeValue()

    def observeValue(self):
        if self.delta > 0:
            return min(self.value, self.stop)
        else:
            return max(self.value, self.stop)


class Trainer(object):
    def __init__(self):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make('BeamRider-v4')
        self.env = wrap_dqn(self.env)

        self.policy_net = DQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        if torch.cuda.device_count() > 1:
            self.policy_net = nn.DataParallel(self.policy_net)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_tracker = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.memory = ReplayMemory(self.params['replay_size'])
        # self.memory = Memory(self.params['replay_size'], 0.6)
        self.beta_scheduler = LinearScheduler(0.4, 1.0, timespan=self.params['epsilon_frames'])
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
        if done:
            self.memory.add(self.state, action, torch.tensor([reward], device=self.device), None, done)
            self.state = self.preprocess(self.env.reset())
        else:
            self.memory.add(self.state, action, torch.tensor([reward], device=self.device), next_state, done)
            self.state = next_state
        return done


    def optimizeModel(self):
        beta = self.beta_scheduler.updateAndGetValue()
        states, actions, rewards, next_states, dones, ISWeights, tree_idx = self.memory.sample(self.batch_size, beta=beta)
        dones = dones.astype(np.uint8)
        states = torch.tensor(states, device=self.device).squeeze(1)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        ISWeights = torch.tensor(ISWeights.astype(np.float32), device=self.device)
        non_final_mask = 1 - torch.tensor(dones, device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([torch.tensor(next_states[idx], device=self.device) for idx, done in enumerate(dones) if not done])
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + rewards
        abs_errors = abs(expected_state_action_values.unsqueeze(1) - state_action_values)
        loss = torch.sum(abs_errors ** 2 * ISWeights) / self.batch_size
        abs_errors_clone = abs_errors.clone().detach().cpu().numpy()
        self.memory.update_priorities(tree_idx, abs_errors_clone + 1e-3)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def prefetch(self):
        while len(self.memory) < self.params['replay_initial']:
            self.addExperience()
        self.score = 0
        print('Done Prefetching.')

    def train(self):
        self.prefetch()
        frame_idx = 0
        for episode in range(3000):
            while 1:
                frame_idx += 1
                # play one move
                game_over = self.addExperience()

                self.optimizeModel()
                if frame_idx % self.params['target_net_sync'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # is this round over?
                if game_over:
                    self.reward_tracker.add(self.score)
                    self.tb_writer.add_scalar('Mean Score | Frames', self.reward_tracker.meanScore(), frame_idx)
                    self.tb_writer.add_scalar('Score | Frames', self.score, frame_idx)
                    self.tb_writer.add_scalar('Epsilon | Frames', self.epsilon_tracker.currentEpsilon, frame_idx)
                    self.tb_writer.add_scalar('Beta | Frames', self.beta_scheduler.observeValue(), frame_idx)
                    self.tb_writer.add_scalar('Mean Score | Frames', self.reward_tracker.meanScore(), frame_idx)
                    self.tb_writer.add_scalar('Score | Episode', self.score, episode)
                    self.tb_writer.add_scalar('Epsilon | Episode', self.epsilon_tracker.currentEpsilon, episode)
                    self.tb_writer.add_scalar('Beta | Episode', self.beta_scheduler.observeValue(), episode)
                    print('Game: %s Score: %s Mean Score: %s' % (episode, self.score, self.reward_tracker.meanScore()))
                    self.score = 0
                    if frame_idx > self.params['total_frames']:
                        return
                    break


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







