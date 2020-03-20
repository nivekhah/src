from envs.ec.component import EdgeServer, TCC
from envs.ec.modify_yaml import ModifyYAML

import numpy as np
import os
import copy


class ECMA(object):
    def __init__(self,
                 seed=None,
                 max_steps=20,
                 bandwidth=[5, 3, 1],
                 cc=104,
                 cl=2.6,
                 n_agents=4,
                 n_actions=2,
                 observation_size=2,
                 prob=[0.3, 0.7, 1],
                 sum_d=644.84,
                 task_proportion=[0.26174748, 0.23825252, 0.24118939, 0.25881061],
                 test_mode=False,
                 model_step=0,
                 state_file=""):
        self.config = ModifyYAML("/home/csyi/pymarl/src/config/envs/ec.yaml")
        self.n_agents = n_agents
        self.bandwidth = bandwidth
        self.cl = cl
        self.cc = cc
        self.prob = prob
        self.n_actions = n_actions
        self.observation_size = observation_size
        self.MAX_STEPS = max_steps
        self.sum_d = sum_d
        self.task_proportion = task_proportion

        self.gen_components()
        self.episode_limit = self.MAX_STEPS
        self.cnt = 0

        self.test_mode = test_mode
        if self.test_mode:
            # file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "train_state.txt")
            self.total_state = np.loadtxt(state_file).tolist()[:model_step]
            self.obs_num = 0
            self.state_num = 0

    def gen_components(self):
        """
        初始化 edge server和 TCC
        :return:
        """
        self.tcc = TCC(self.cc)
        self.edge_servers = []
        for i in range(self.n_agents):
            self.edge_servers.append(EdgeServer(i, self.cl, self.prob, self.bandwidth))

    def step(self, actions):
        self.cnt += 1
        T = self.do_actions(actions)  # 处理完任务所花费的时间
        # print("处理完任务所花的总时间", T)
        if self.cnt == self.MAX_STEPS:
            done = True
        else:
            done = False
        reward = self.sum_d / T
        if not done:
            self.ready_for_next_step()
        # print(reward)
        return reward, done, {}

    def ready_for_next_step(self):
        tasks = self.distribute_task()
        for es in self.edge_servers:
            es.next_step(tasks[es.id])

    def do_actions(self, actions):
        """
        执行对应的 action，返回相应的处理时间
        :param actions:
        :return:
        """
        T = []
        for es in self.edge_servers:
            time = es.do_action(actions[es.id], self.tcc)
            # print("执行动作: ", actions[es.id], "所发费的时间为", time)
            T.append(time)
        return np.max(T)

    def get_obs(self):
        if self.test_mode:
            # 将 state 拆分成 self.n_agents 个 observation
            agents_obs = []
            s = self.total_state[self.obs_num]
            for i in range(self.n_agents):
                temp_obs = np.array([s[i * 2 + 1], s[i * 2]])
                agents_obs.append(temp_obs)
            self.obs_num += 1
        else:
            agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        obs = self.edge_servers[agent_id].get_obs()
        return obs

    def get_obs_size(self):
        return self.observation_size

    def get_state(self):
        """
        获取环境的全局状态，即将每一个 edge server 的 observation 拼凑起来。
        :return: 全局状态
        """
        if self.test_mode:
            state = self.total_state[self.state_num]
            self.state_num += 1
        else:
            state = []
            for es in self.edge_servers:
                state.append(es.b)
                state.append(es.d)
        return np.array(state)

    def get_state_size(self):
        """
        获取全局状态的大小，即全局状态的大小等于每一个 edge server 的 observation 大小的累和。
        :return: 全局状态的大小
        """
        size = self.observation_size * self.n_agents
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self.cnt = 0
        self.tcc.reset()
        tasks = self.distribute_task()
        for es in self.edge_servers:
            es.reset(tasks[es.id])

    def distribute_task(self):
        tasks = []
        for item in self.task_proportion:
            tasks.append(self.sum_d * item)
        return tasks

    def render(self):
        print("ec::render")

    def close(self):
        print("ec::close")

    def seed(self):
        print("ec::seed")

    def save_replay(self):
        print("ec::replay")

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), "output", "train_state.txt")
    env = ECMA(seed=None,
               max_steps=20,
               bandwidth=[2, 1, 0.1],
               cc=40,
               cl=1,
               n_agents=4,
               n_actions=2,
               observation_size=2,
               prob=[0.8, 0.9, 1],
               sum_d=10,
               task_proportion=[0.25, 0.25, 0.25, 0.25],
               test_mode=True,
               model_step=10,
               state_file=file_path)
    env.reset()
    for i in range(10):
        print("##############", i, "################")
        print(env.get_obs())
        print(env.get_state())
