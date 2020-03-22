from envs.ec.ec_env import ECMA
import numpy as np
from path import Path
from envs.ec.optimal_qmix import OptimalQMIX
import copy

import os

"""
Policy 类包括四种算法：随机算法, 全部 local, 全部 offload, 最优 QMIX 算法 
"""


class Policy:

    def __init__(self, env: ECMA, policy: str):
        self.__env = env
        self.__episodes_reward = []
        self.__agents = []
        self.__n_agents = env.n_agents
        self.__policy = policy
        self.__reward_list = []
        self.__state_list = []
        self.__action_list = []

        self.gen_agent()

    def run(self, t_max):
        episodes = int(t_max / self.__env.MAX_STEPS)  # 计算出总共需要执行的回合数
        for i in range(episodes):
            self.__env.reset()  # 初始化环境
            episode_reward = 0  # 每一个回合的累积总 reward
            for j in range(self.__env.MAX_STEPS):
                state = self.__env.get_state()  # 从环境获取一个状态
                self.__state_list.append(state.tolist())  # 将得到 state 保存下来
                obs = self.__env.get_obs()  # 获取 observations

                actions = []
                for index, agent in enumerate(self.__agents):
                    action = agent.select_action(obs[index])
                    actions.append(action)
                self.__action_list.append(actions)
                reward, done, _ = self.__env.step(actions)

                self.__reward_list.append(reward)
                episode_reward += reward
            self.__episodes_reward.append(episode_reward)

    def gen_agent(self):
        if self.__policy == "all_offload":
            for i in range(self.__n_agents):
                self.__agents.append(AllOffloadAgent())
        elif self.__policy == "all_local":
            for i in range(self.__n_agents):
                self.__agents.append(AllLocalAgent())
        elif self.__policy == "random":
            for i in range(self.__n_agents):
                self.__agents.append(RandomAgent())
        elif self.__policy == "optimal":
            for i in range(self.__n_agents):
                self.__agents.append(OptimalAgent())

    @property
    def episodes_reward(self):
        return copy.deepcopy(self.__episodes_reward)

    @property
    def total_state(self):
        return copy.deepcopy(self.__state_list)

    @property
    def total_reward(self):
        return copy.deepcopy(self.__reward_list)

    @property
    def total_action(self):
        return copy.deepcopy(self.__action_list)


class AllOffloadAgent:

    def __init__(self):
        pass

    @staticmethod
    def select_action(obs):
        """
        All Offload 算法选择全部上传
        :return:
        """
        return 1


class AllLocalAgent:

    def __init__(self):
        pass

    def select_action(self, obs):
        """
        All Local 算法选择全部本地执行
        :return:
        """
        return 0


class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, obs):
        """
        Random 算法 action 的选择完全随机
        :return:
        """
        return np.random.randint(0, 2)


class OptimalAgent:
    """
    执行最优算法的
    """
    def __init__(self):
        self.__optimal_qmix = OptimalQMIX(os.path.join(Path.get_envs_config_path(), "ec.yaml"))

    def select_action(self, obs):
        """
        最优 QMIX 算法通过最优算法选择最优 action
        :param obs:
        :return:
        """
        return self.__optimal_qmix.select_optimal_action(obs)


if __name__ == '__main__':
    env = ECMA()
    policy = Policy(env, "random")
    policy.run(2000)
    statistic = [{}, {}, {}, {}]
    for action in policy.total_action:
        for i in range(len(action)):
            if action[i] in statistic[i].keys():
                statistic[i][action[i]] = statistic[i][action[i]] + 1
            else:
                statistic[i][action[i]] = 1
    print(statistic)

