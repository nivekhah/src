from envs.multiagentenv import MultiAgentEnv
from envs.wsn.Component import Satellite, Sensor, BaseStation
from envs.wsn.Configuration import config
import numpy as np
import copy

class Environment(MultiAgentEnv):
    def __init__(self,state_last_action=True,seed=None):
        self.connections = config.get("connections")
        self.n_agents = config.get("sensor_num")
        self.base_station = BaseStation()
        self.satellite = Satellite()
        self.sensors = [Sensor(i) for i in range(self.n_agents )]
        self.n_actions = config.get("n_actions")
        self.observation_size = config.get("observation_size")
        self.state_last_action = config.get("state_last_action")
        self.last_action = np.zeros((self.n_agents, self.n_actions)) ##
        self.MAX_TIMES = config.get("MAX_TIMES") ##
        self.decision_interval = config.get("decision_interval")
        self.sample_rate = config.get("sample_rate")
        self.episode_limit = config.get("MAX_TIMES") #最大step数量，同时也为buffer中的宽度，episode_limit+1
        self.end_obs = np.array([-1]*self.observation_size)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        # print("obss: ", agents_obs)
        return agents_obs

    def get_total_actions(self):
        return self.n_actions

    def get_obs_agent(self, agent_id):
        """
        获取单个 agent 的 observation
        :param agent_id:
        :return: 
        """
        if self.cnt == self.MAX_TIMES:
            return self.end_obs
        obs = 0
        for sensor1 in self.sensors:
            if sensor1.id == agent_id:
                #传参数给get_observation，告知其周围的连接
                components = []
                connection = self.connections[agent_id]
                for component_id in connection:
                    if component_id == self.satellite.id:
                        component = self.satellite
                        components.append(component)
                    elif component_id == self.base_station.id:
                        component = self.base_station
                        components.append(component)
                    else:
                        for sensor in self.sensors:
                            if sensor.id == component_id:
                                component = sensor
                                components.append(component)
                obs = sensor1.get_observation(components)
                # print("state:", obs)
        return copy.deepcopy(obs)

    def step(self, actions):
        '''
        :param actions:         [1,1,1]
        :return: 
        '''
        # former_cache = self.base_station.cache + self.satellite.cache
        self.cnt += 1
        action_results = self.do_actions(actions)
        # rewards = []
        # for i in range(self.n_agents):
        #     if not action_results[i]:
        #         reward = self.punishment
        #     else:
        #         dif_cache = self.base_station.cache + self.satellite.cache - former_cache
        #         reward = dif_cache / (self.decision_interval * self.sample_rate * self.n_agents * self.MAX_TIMES)
        #     rewards.append(reward)

        actions = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions)]
        if self.cnt == self.MAX_TIMES:
            done = True
        else:
            done = False

        reward = 0
        if done:  # 如果此回合结束，将 基站和卫星接收的总数据量/sensor收集的总数据量 作为总的reward
            reward = (self.base_station.cache + self.satellite.cache) / \
                     (self.decision_interval * self.sample_rate * self.MAX_TIMES * self.n_agents)
        else:
            reward = 0

        for sensor in self.sensors:
            sensor.sample_data()
        return reward, done, {}

    def get_avail_agent_actions(self, agent_id):
        '''
        returns the available actions for agent_id.
        :param agent_id: 
        :return: 
        '''
        return [1] * self.n_actions

    def get_obs_size(self):
        return self.observation_size

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_state_size(self):
        size = self.n_agents*1+1+1
        if self.state_last_action:
            size += self.n_agents * self.n_actions
        return size


    def reset(self):
        self.cnt = 0  ##
        self.base_station.reset()
        self.satellite.reset()
        for sensor in self.sensors:
            sensor.reset()
        for sensor in self.sensors:
            sensor.sample_data()
        self.last_action = np.zeros((self.n_agents, self.n_actions))

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        base_state_list = [self.base_station.cache,self.satellite.cache]
        sensors_state_list = [sensor.cache for sensor in self.sensors]
        base_state_list.extend(sensors_state_list)
        state = np.array(base_state_list)
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        return state

    def get_env_info(self):
        return super().get_env_info()

    def do_actions(self, actions):
        action_results = []
        for i in range(self.n_agents):
            components = []
            connection = self.connections[self.sensors[i].id]
            for component_id in connection:
                if component_id == self.satellite.id:
                    component = self.satellite
                    components.append(component)
                elif component_id == self.base_station.id:
                    component = self.base_station
                    components.append(component)
                else:
                    for sensor in self.sensors:
                        if sensor.id == component_id:
                            component = sensor
                            components.append(component)
            action_result = self.sensors[i].do_action(actions[i], components)
            action_results.append(action_result)
        for i in range(self.n_agents):
            self.sensors[i].do_settlement()
        return action_results

    def seed(self):
        print("env::seed")

    def close(self):
        print("env::save_replay")

    def render(self):
        print("env::reder")

    def save_replay(self):
        print("env::save_replay")