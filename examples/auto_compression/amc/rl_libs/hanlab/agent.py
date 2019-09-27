# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from .memory import SequentialMemory
from .utils import to_numpy, to_tensor
from copy import deepcopy

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = self.fc11(x) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.warmup = args.warmup

        #
        self.epsilon = 1.0
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)  # save to memory
            # self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        return action

    def select_action(self, s_t, episode):
        # assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
        action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)
        delta = self.init_delta * (self.delta_decay ** (episode - self.warmup))
        # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action, sigma=delta)
        action = np.clip(action, self.lbound, self.rbound)

        # self.a_t = action
        return action

    def reset(self, obs):
        pass
        # self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        from scipy import stats
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


def train(num_episode, agent, env, output, warmup):
    text_writer = open(os.path.join(output, 'sh_log.txt'), 'w')
    print('=> Output path: {}...'.format(output))

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        env.render()
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        #if episode % int(num_episode / 3) == 0:
         #   agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            final_reward = T[-1][0]
            # print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > warmup:
                    agent.update_policy()

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            # tfwriter.add_scalar('reward/last', final_reward, episode)
            # tfwriter.add_scalar('reward/best', env.best_reward, episode)
            # tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            # tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            # tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            # # record the preserve rate for each layer
            # for i, preserve_rate in enumerate(env.strategy):
            #     tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            #text_writer.write('best policy: {}\n'.format(env.best_strategy))
    text_writer.close()
