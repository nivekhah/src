from src.envs.ec.config import config
from src.envs.ec.component import EdgeServer, TCC
import numpy as np
class ECMA(object):
    def __init__(self,seed=None):
        self.n_agents = config.get("n_agents")
        self.gen_components()
        self.n_actions = config.get("n_actions")
        self.observation_size = config.get("observation_size")
        self.MAX_STEPS = config.get("MAX_STEPS")
        self.episode_limit = self.MAX_STEPS
        self.cnt = 0

    def gen_components(self):
        '''
        初始化edge server和TCC
        :return:
        '''
        cl = config.get("cl")
        cc = config.get("cc")
        self.tcc = TCC(cc)
        self.edge_servers = []
        for i in range(self.n_agents):
            self.edge_servers.append(EdgeServer(i,cl))






    def step(self, actions):
        self.cnt += 1
        T = self.do_actions(actions) ##处理完任务所花费的时间
        if self.cnt == self.MAX_STEPS:
            done = True
        else:
            done = False
        reward = self.sum_d/T
        if not done:
            self.ready_for_next_step()
        # print(reward)
        return reward, done, {}


    def ready_for_next_step(self):
        tasks = self.distribute_task()
        for es in self.edge_servers:
            es.next_step(tasks[es.id])


    def do_actions(self,actions):
        '''
        执行对应的action，返回相应的处理时间
        :param actions:
        :return:
        '''
        T = []
        for es in self.edge_servers:
            time = es.do_action(actions[es.id],self.tcc)
            T.append(time)
        return np.max(T)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        obs = self.edge_servers[agent_id].get_obs()
        return obs

    def get_obs_size(self):
        return self.observation_size

    def get_state(self):
        '''
        es的任务量及带宽
        :return:
        '''
        state = []
        for es in self.edge_servers:
            state.append(es.b)
            state.append(es.d)
        return np.array(state)

    def get_state_size(self):
        size = self.observation_size*self.n_agents
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
        sum_d = config.get("sum_d")
        self.sum_d = sum_d
        task_proportion = config.get("task_proportion")
        for item in task_proportion:
            tasks.append(sum_d*item)
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
