#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)#内部创建一个环境，与测试环境分开；
        self.state = self.env.reset()#angent内部变量，用来记录当前所处的状态；
        self.rewards = collections.defaultdict(float)#奖励：s1,a,s2->r
        self.transits = collections.defaultdict(collections.Counter)#s1,a->(s2:n)
        self.values = collections.defaultdict(float)#value table

    #随机玩count步；
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()#随机取一个action
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward#记录历史的奖励获取情况；
            self.transits[(self.state, action)][new_state] += 1#记录迁移历史情况；
            self.state = self.env.reset() if is_done else new_state#更新当前状态；

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0#如果之前经验中没有(state,action)的记录，则返回的动作值为0；
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            #count/total:计算的是从状态s，执行动作a后，迁移到状态s'的概率；
            #self.values：假装我们已经拥有了目标状态tgt_state的状态值；问题是，迭代的原理到底是什么？这是Bellman证明的关键；
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    #在输入状态state下，对每个动作进行循环，计算每个动作的动作值，并把最大的动作返回；这里没有直接用到状态值，而是通过cal_action_value进行了动态计算；
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    #根据当前的value table来玩一个回合的游戏；
    #测试环境env;与agent内部的环境区分开来；
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            #基于现有的状态值的表进行动作的筛选；
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward#在测试的时候，也同时积累经验；
            self.transits[(state, action)][new_state] += 1#在测试的时候，也同时积累经验
            total_reward += reward#计算当前回合的总收益；
            if is_done:
                break
            state = new_state
        return total_reward

    #value table的迭代：计算各个状态的值；
    def value_iteration(self):
        #对每个状态进行循环，分别计算他们的值；
        for state in range(self.env.observation_space.n):
            #对动作进行循环，把当前状态state下的动作action的动作值进行计算，得到了动作值的列表；
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)#把当前状态state的值记录下来；其实就是更新state的状态值


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()

    import time
    video_base_dir = "D:\\deepLearning\\videos\\"
    board_base_dir = "D:\\deepLearning\\tensorboard\\"
    file_name = "ch05_01_frozenlake_v_iteration"
    t = time.time()
    video_dyn_dir = video_base_dir+file_name+str(int(t))
    board_dyn_dir = board_base_dir+file_name+str(int(t))
    env = gym.wrappers.Monitor(test_env, directory=video_dyn_dir)
    writer = SummaryWriter(log_dir=board_dyn_dir, comment="-v-iteration")


    iter_no = 0#状态值的迭代次数；
    best_reward = 0.0#当前为止获得的最佳收益
    while True:
        iter_no += 1
        agent.play_n_random_steps(1000)#随机玩100步；玩的步骤很少，则无法获取足够的经验；经验越多，收敛越快
        agent.value_iteration()#根据回合记录的历史情况来更新value table；不断更新每个状态的值；

        reward = 0.0
        for _ in range(TEST_EPISODES):#测试回合：玩20个测试的回合；
            reward += agent.play_episode(env)#记录这二十个测试回合的中的收益
        reward /= TEST_EPISODES#计算二十个测试回合的平均收益
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:#更新最佳收益
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:#如果二十个回合的平均收益达到0.8，则算是解决了这个问题；
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
