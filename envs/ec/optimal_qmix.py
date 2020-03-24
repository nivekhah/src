"""
OptimalQMIX 类用于获得 TCC, Edge Server 场景下， Edge Server 对应于某一个 Observation 的最优 Action，以及最优 Reward。
"""

from envs.ec.modify_yaml import ModifyYAML
import copy


class OptimalQMIX:
    def __init__(self, config_path):
        self.__config_path = config_path
        self.__cl = None  # 本地 edge server 的计算能力
        self.__cc = None  # TCC 的计算能力
        self.__observation_size = None  # 每一个 agent 的观测值的大小
        self.__n_agents = None  # agent 个数
        self.__n_actions = None  # 每一个 agent 允许的动作数
        self.__sum_task = None  # 任务总量
        self.__parse_config()

    def get_optimal_from_state(self, state: list):
        """
        :param state: 当前环境的全局状态 [bandwidth_1, task_1, bandwidth_2, task_2, ...]
        :return: 返回全局状态对应最优 action 和最优 reward
        """
        optimal_actions = []  # 每一个 agent 的最优动作
        agent_times = []  # 每一个 agent 在选择最优动作时的执行时间
        for i in range(int(len(state) / self.__observation_size)):
            task = state[i*self.__observation_size + 1]
            bandwidth = state[i*self.__observation_size]
            local_time = self.__do_local(task)
            offload_time = self.__do_offload(task, bandwidth)
            if local_time >= offload_time:
                optimal_actions.append(1)
                agent_times.append(offload_time)
            else:
                optimal_actions.append(0)
                agent_times.append(local_time)
        optimal_reward = self.__sum_task / max(agent_times)  # 全局状态 state 对应的最优 reward
        return copy.deepcopy(optimal_actions), optimal_reward

    def select_optimal_action(self, obs):
        task = obs[0]
        bandwidth = obs[1]
        t_local = self.__do_local(task)
        t_offload = self.__do_offload(task, bandwidth)
        if t_local >= t_offload:
            return 1
        else:
            return 0

    def __do_local(self, task: float):
        """
        :param task: 任务量
        :return: 返回任务在本地执行的时间
        """
        return task / self.__cl

    def __do_offload(self, task: float, bandwidth: float):
        """
        :param task: 任务量
        :param bandwidth: 带宽
        :return: 返回任务 offload 到 TCC 执行的时间
        """
        tran_time = task / bandwidth
        c_time = task / self.__cc
        return tran_time + c_time

    def __parse_config(self):
        """
        解析 ec.yaml 文件中的环境配置项
        :return:
        """
        env_conf = ModifyYAML(self.__config_path).data["env_args"]
        self.__cc = env_conf["cc"]
        self.__cl = env_conf["cl"]
        self.__n_agents = env_conf["n_agents"]
        self.__n_actions = env_conf["n_actions"]
        self.__observation_size = env_conf["observation_size"]
        self.__sum_task = env_conf["sum_d"]

