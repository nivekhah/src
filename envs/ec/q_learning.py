#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'pymarl'
@author  : 'zzw'
@file    : 'q_learning'.py
@ide     : 'PyCharm'
@time    : '2020'-'03'-'22' '15':'11':'35'
@contact : zongwang.zhang@outlook.com
'''
import numpy as np
import pandas as pd
import copy
import os
import time
import json
from matplotlib import pyplot as plt
from src.envs.ec.data_processor import DataSaver
pd.set_option('precision', 7)
class QLearningTable:
    param = {
        "learning_rate":0.01,
        "reward_decay":0.9,
        # "epsilon_anneal_time": 400000,
        "epsilon_finish": 0.1,
        # "epsilon_start": 1.0,
        # "save_model": True,
        # "save_model_interval": 5000,
        "model_path":"",
        "n_agent":4,
        "n_action":2,
        "use_model":False, #设置为真的时候，同时要设置model_path
        # "test_interval":1000,
        # "test_mode":True
    }

    def get_epsilon(self,):
        """
        获取epsilon
        :return:
        """
        return self.param["epsilon_finish"]

    # 初始化
    def __init__(self):
        self.n_action = self.param["n_action"]
        self.n_agent = self.param["n_agent"]
        self.actions = list(range(self.n_action**self.n_agent)) # a list
        self.lr = self.param["learning_rate"] #学习率
        self.gamma = self.param["reward_decay"]  # 奖励衰减
        #加载Q table
        self.use_model = self.param["use_model"]
        if self.use_model:
            self.q_table = self.load_model(self.param["model_path"],self.actions)
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def load_model(self,model_path,actions):
        """
        从文件中读取模型（Q表）
        :param model_path 模型的具体地址
        :param actions 动作空间
        :return:
        """
        if not os.path.exists(model_path):
            print("model path not exist!")
            return
        q_table = pd.read_csv(model_path, index_col=0)
        q_table.columns = actions
        return q_table


    def choose_action(self, observation):
        """
        选择动作（epsilon)
        :param observation:
        :return:
        """
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在
        # 选择 action
        if np.random.uniform() < 1 - self.get_epsilon():  # 选择 Q value 最高的 action
            state_action = self.q_table.loc[observation, :]
            action_list = copy.deepcopy(state_action)
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(action_list[action_list == np.max(action_list)].index)
        else:  # 随机选择 action
            ##这个地方加一个限制 只能选择未做的任务
            state_action = self.q_table.loc[observation, :]
            action_list = copy.deepcopy(state_action)
            action = np.random.choice(action_list.index)
        # print(refined_action)
        return action



    # 学习更新参数
    def learn(self, s, a, r, s_,done):
        """
        按照Q-learning更新公式对Q值进行更新
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        """
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.loc[s, a]

        assert isinstance(s_,str)
        temp = copy.copy(s_)
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 下个 state 不是 终止符
        else:#terminal
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)# 更新对应的state-action值


    # 检测 state 是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        # print(self.q_table)


    def save_model(self):
        """
        保存模型
        :return:
        """
        dir = os.path.join(os.getcwd(),"data","q_model")
        if not os.path.exists(dir):
            os.makedirs(dir)
        model_path = os.path.join(dir,"q_tabel_"+time.strftime("%Y-%m-%d_%H-%M-%S")+".csv")
        self.q_table.to_csv(model_path)
        print(model_path)


    def refine_action(self,action):
        r = action
        refined_action = []
        refined_action.append(int(r%self.n_action))
        r = int(r / self.n_action)
        while r >= self.n_action:
            refined_action.append(r % self.n_action)
            r = int(r / self.n_action)
        refined_action.append(r)
        for _ in range(self.n_agent-len(refined_action)):
            refined_action.append(0)
        refined_action.reverse()
        return refined_action


def train(times=10000):
    data_saver = DataSaver("RL_training")
    data_saver.add_item("start_time", time.strftime("%Y-%m-%d_%H-%M-%S"))
    from src.envs.ec.ec_env import ECMA
    env = ECMA()
    RL = QLearningTable()
    data_saver.add_item("RL_param",RL.param)
    data_saver.add_item("episodes",times)
    for episode in range(0, times):
        env.reset()
        observation = env.get_obs()
        observation = list(np.round(np.array(observation,dtype=float).flatten(), 3))
        while True:
            action = RL.choose_action(str(observation))
            refined_action = RL.refine_action(action)
            reward, done ,_ = env.step(refined_action)
            observation_ = env.get_obs()
            observation_ = list(np.round(np.array(observation_,dtype=float).flatten(), 3))
            RL.learn(str(observation), action, reward, str(observation_),done)
            observation = copy.deepcopy(observation_)
            data_saver.append(str(episode),[observation,refined_action,reward])
            if done:
                if episode / 1000 > 0:
                    print(episode, time.strftime("%Y-%m-%d_%H-%M-%S"))
                if episode / 10000:
                    RL.save_model()
                break
    data_saver.add_item("end_time", time.strftime("%Y-%m-%d_%H-%M-%S"))
    data_saver.to_file()

def test(times):
    data_saver = DataSaver("qlearning_test")
    data_saver.add_item("start_time", time.strftime("%Y-%m-%d_%H-%M-%S"))
    from src.envs.ec.ec_env import ECMA
    env = ECMA()
    RL = QLearningTable()
    data_saver.add_item("qlearning_param",RL.param)
    data_saver.add_item("episodes",times)
    for episode in range(0, times):
        env.reset()
        observation = env.get_obs()
        observation = list(np.round(np.array(observation,dtype=float).flatten(), 3))
        while True:
            action = RL.choose_action(str(observation))
            refined_action = RL.refine_action(action)
            reward, done ,_ = env.step(refined_action)
            observation_ = env.get_obs()
            observation_ = list(np.round(np.array(observation_,dtype=float).flatten(), 3))
            observation = copy.deepcopy(observation_)
            data_saver.append(str(episode),[observation,refined_action,reward])
            if done:
                if episode / 1000 > 0:
                    print(episode, time.strftime("%Y-%m-%d_%H-%M-%S"))
                break
    data_saver.add_item("end_time", time.strftime("%Y-%m-%d_%H-%M-%S"))
    data_saver.to_file()



def extract_reward(filename="RL_training_2020-03-23_08-26-58"):
    """
    从训练文件中提取reward保存为文件
    :return:
    """
    training_file_dir = os.path.join(os.getcwd(),"data","RL_training")
    traning_file_path = os.path.join(training_file_dir,filename)
    data = json.load(open(traning_file_path,"r"))
    reward = []
    episodes = data["episodes"]
    for i in range(episodes):
        episode_record = data[str(i)]
        for step_record in episode_record:
            reward.append(step_record[2])
    # print("reward",reward)
    reward_file_path = os.path.join(training_file_dir,"reward"+filename[11:])
    np.savetxt(reward_file_path,np.array(reward))

def plot_training(filename="RL_training_2020-03-23_08-26-58"):
    training_file_dir = os.path.join(os.getcwd(), "data", "RL_training")
    reward_file_path = os.path.join(training_file_dir, "reward" + filename[11:])
    reward = np.loadtxt(reward_file_path)
    reward = reward.reshape(-1,1000)
    reward = np.mean(reward,1)
    plt.plot(list(range(len(reward))),reward)
    plt.show()

def verify_state():
    """
    读取Q-table，判断那些状态没有收敛
    :return:
    """
    pass

def plot_state_training():
    """
    画出状态的训练状况
    :return:
    """
    pass

def plot_optimal_training():
    """
    画出每个状态训练次数和达到最优的次数，以及比例，给出图和表格
    :return:
    """
    pass




if __name__ == '__main__':
    # train(50000)
    # RL = QLearningTable()
    # RL.refine_action(4)
    # extract_reward()
    plot_training()