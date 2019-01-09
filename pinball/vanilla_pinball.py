#!/usr/bin/env python3

import gym
import csv
from tensorboardX import SummaryWriter
import copy
import sys
sys.path.insert(0, '..')
from DQN import *
from wrappers import wrap_dqn
import argparse
import datetime


import os; os.environ["CUDA_VISIBLE_DEVICES"]="1"
from DQN import HYPERPARAMS



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
        self.env = gym.make('AtlantisNoFrameskip-v4')
        self.env = wrap_dqn(self.env)
        self.policy_net = Dueling_DQN(self.env.observation_space.shape, self.env.action_space.n, self.device).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.reward_tracker = RewardTracker()
        self.episode = 0
        self.state = self.preprocess(self.env.reset())
        self.score = 0
        self.batch_size = HYPERPARAMS['batch_size']

        csv_file = open('vanilla_dqn.csv', 'w+')
        self.writer = csv.writer(csv_file)


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
        start = datetime.datetime.now()
        while True:
            game_over = self.addExperience()
            if game_over:
                self.reward_tracker.add(self.score)
                print('Game: %s Score: %s Mean Score: %s' % (self.episode, self.score, self.reward_tracker.meanScore()))
                self.score = 0

            # are we done prefetching?
            if len(self.policy_net.memory) < HYPERPARAMS['replay_initial']:
                continue

            self.policy_net.optimizeModel(self.target_net)
            time_delta = (datetime.datetime.now() - start).total_seconds()
            self.writer.writerow([self.episode, self.score, self.reward_tracker.meanScore(), self.policy_net.epsilon_tracker._epsilon, time_delta])
            if self.episode >= HYPERPARAMS['episodes']:
                return



    # def playback(self, path):
    #     target_net = torch.load(path, map_location='cpu')
    #     env = gym.make('PongNoFrameskip-v4')
    #     env = wrap_dqn(env)
    #     state = self.preprocess(env.reset())
    #     done = False
    #     score = 0
    #     import time
    #     while not done:
    #         time.sleep(0.015)
    #         env.render(mode='human')
    #         action = torch.argmax(target_net(state), dim=1).to(self.device)
    #         state, reward, done, _ = env.step(action.item())
    #         state = self.preprocess(state)
    #         score += reward
    #     print("Score: ", score)



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






