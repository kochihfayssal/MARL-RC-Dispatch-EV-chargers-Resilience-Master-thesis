import numpy as np
import os
from Episode_rollout import RolloutWorker
from agent import Agents
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.args = args

        # self.win_rates = []
        self.episode_rewards = []

        self.save_path = self.args.result_dir + '/' + args.alg+ '/' + "test" + "/" + "test1"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num=0):

        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            # if self.args.anneal_lr:
            #     frac = 1.0 - time_steps / self.args.n_steps
            #     lrnow = frac * self.args.learning_rate
            #     self.agents.policy.ac_optimizer.param_groups[0]["lr"] = lrnow

            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                # self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(time_steps)
                evaluate_steps += 1
            episodes = []

            for episode_idx in range(self.args.n_episodes):
                episode, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.agents.train(episode_batch, train_steps, time_steps)
            train_steps += self.args.ppo_n_epochs

        episode_reward = self.evaluate()
        # print('win_rate is ', win_rate)
        # self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(time_steps)

    def evaluate(self):
        # win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            # if win_tag:
                # win_number += 1
        return episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure(figsize=(12, 8))
        plt.axis([0, self.args.n_steps, 0, 5000])
        plt.cla()

        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('1e4 timesteps')
        # plt.ylabel('win_rate')

        # plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('1e4 timesteps')
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        # np.save(self.save_path + '/win_rates'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)

        plt.close()