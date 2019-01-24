#!/usr/bin/env python3

import gym
from tensorboardX import SummaryWriter
import copy
import sys
sys.path.insert(0, '..')
from DQN import *
from wrappers import wrap_dqn
import argparse
import datetime
from DQN import HYPERPARAMS
import os

NUM_EPISODES = 700


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


class Trainer(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make('PongNoFrameskip-v4')
        self.env = wrap_dqn(self.env)
        self.policy_net = DQN(self.env.observation_space.shape, self.env.action_space.n, self.device).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.reward_tracker = RewardTracker()
        self.episode = 0
        self.state = self.preprocess(self.env.reset())
        self.score = 0
        self.batch_size = HYPERPARAMS['batch_size']
        self.tb_writer = SummaryWriter('results_pong')


    def preprocess(self, state):
        state = torch.tensor(np.expand_dims(state, 0)).to(self.device)
        return state.float() / 256


    def addExperience(self):
        action = self.policy_net.getAction(self.state)
        next_state, reward, done, _ = self.env.step(action.item())
        self.score += reward
        if done:
            self.policy_net.memory.push(self.state, action, torch.tensor([reward], device=self.device), None)
            self.state = self.preprocess(self.env.reset())
            self.episode += 1
        else:
            next_state = self.preprocess(next_state)
            self.policy_net.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state)
            self.state = next_state
        return done


    def train(self):
        total_steps = 0
        for episode in range(NUM_EPISODES):
            for iteration in range(100000000):
                total_steps += 1
                game_over = self.addExperience()

                # are we done prefetching?
                if len(self.policy_net.memory) < HYPERPARAMS['replay_initial']:
                    continue
                if len(self.policy_net.memory) == HYPERPARAMS['replay_initial']:
                    self.score = 0
                    self.state = self.preprocess(self.env.reset())
                    break

                if game_over:
                    self.reward_tracker.add(self.score)
                    self.tb_writer.add_scalar('Score', self.score, episode)
                    self.tb_writer.add_scalar('Average Score', self.reward_tracker.meanScore(), episode)
                    self.tb_writer.add_scalar('Steps per episode', iteration, episode)
                    print('Game: %s Score: %s Mean Score: %s' % (self.episode, self.score, self.reward_tracker.meanScore()))
                    self.score = 0

                self.policy_net.optimizeModel(self.target_net)



if __name__ == "__main__":
    # set up which GPU to use
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()
    gpu_num = args.gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    # random seeds
    np.random.seed(5)
    random.seed(5)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)

    trainer = Trainer()
    print('Trainer Initialized')
    trainer.train()

