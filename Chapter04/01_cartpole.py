#!/usr/bin/env python3
import time

import gym
from collections import namedtuple
import numpy as np
import uuid as uuid
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 32
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    #x为在可调用的实例时传入的输入参数。每个模块都要自己定义这个数据转换的方式，这里最简单的实现就是直接将其纳入到神经网络
    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

#一直循环生成batch，即16个回合的数据。没有停止条件，只有靠外层调用的代码去结束这个循环，即break;
def iterate_batches(env, net, batch_size):
    batch = []#存储这个batch里面所有的回合。这里设置的值是16个回合；
    episode_reward = 0.0#每个回合所获得的总体的奖励
    episode_steps = []#每个回合的每一步的记录
    obs = env.reset()#每次都重置环境
    sm = nn.Softmax(dim=1)#在输入参数的第二维度进行softmax计算，对每一个第二维度的列表的值进行这个转换
    while True:#一直循环，直到跑完
        obs_v = torch.FloatTensor([obs])#将环境的tensor转换成torch能接受的格式；
        netResult = net(obs_v);#其实这里的net在一直进化，返回的动作的概率分布也不一样
        act_probs_v = sm(netResult)#得到每个动作的概率；shape和netResult一致
        act_probs = act_probs_v.data.numpy()[0]#这里假设输出的结果的维度是[1,2]；取出每个action的概率值；
        action = np.random.choice(len(act_probs), p=act_probs)#根据每个动作的概率分布，来随机获取下一个需要进行的action;概率值大的动作则有很大概率被选中；
        next_obs, reward, is_done, _ = env.step(action)#将选中的动作作用于环境；
        episode_reward += reward#将本回合内的奖励值累加；
        episode_steps.append(EpisodeStep(observation=obs, action=action))#将当前这一步加入到回合步骤的列表中；不用最新的next_obs，而是用当前的obs;
        if is_done:#如果回合结束，则：1、将该回合加入到batch中；2、将该回合奖励清零；3、将该回合步骤清零；4、将环境重置；
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:#如果回合数达到了指定的batch的数量，则返回batch给调用者；即是生成一个batch给调用者；
                yield batch#返回生成的batch，外层调用代码执行完后，再返回到这个方法的时候，接着下一句执行
                batch = []#如果返回了batch，先清空batch
        obs = next_obs#设置下一个观察结果；如果episode结束了，则是重置后的观察结果，如果没有结束，则是env.step操作后返回的观察结果


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean



if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    video_base_dir = "G:\\ai\\videos\\"
    board_base_dir = "G:\\ai\\tensorboard\\"
    file_name = "ch04_01_01_cartpole_"
    t = time.time()
    video_dyn_dir = video_base_dir+file_name+str(int(t))
    board_dyn_dir = board_base_dir+file_name+str(int(t))
    env = gym.wrappers.Monitor(env, directory=video_dyn_dir)
    writer = SummaryWriter(log_dir=board_dyn_dir, comment="-cartpole")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    #调用iterate_batches这个batch生成器；
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
