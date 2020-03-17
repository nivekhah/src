from src.envs.ec.ec_env import ECMA
import numpy as np
import copy

"""
Policy 类包括三种算法： 随机算法， 全部 local， 全部 offload
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
        temp_dict = {}

        episodes = int(t_max / self.__env.MAX_STEPS)
        for i in range(episodes):
            self.__env.reset()
            episode_reward = 0
            for j in range(self.__env.MAX_STEPS):
                state = self.__env.get_state()
                self.__state_list.append(state)
                self.__env.get_obs()

                actions = []
                for agent in self.__agents:
                    action = agent.select_action()
                    actions.append(action)
                self.__action_list.append(actions)
                reward, done, _ = self.__env.step(actions)

                if reward in temp_dict.keys():
                    count = temp_dict[reward]
                    temp_dict[reward] = count + 1
                else:
                    temp_dict[reward] = 1
                # if reward != (10/2.5) and reward != (10/25.25) and reward != (10/2.75) and reward != (10/1.5):
                #     print("异常 reward: ", reward)
                # print("[state]: ", state, "\t[actions]: ", actions, "\t[reward]: ", reward)
                self.__reward_list.append(reward)
                episode_reward += reward
            self.__episodes_reward.append(episode_reward)

        expectation = 0
        for key in temp_dict.keys():
            probability = temp_dict[key] / t_max
            expectation += float(key)*probability
        print(temp_dict, "期望值：", expectation)

    def cal_max_expectation(self):
        measure_state, measure_reward = self.__get_measure_state_reward()

        # if self.__policy is "all_offload":
        #     for index, item in enumerate(measure_state):
        #         print("state: ", item)
        #         print("reward: ", measure_reward[index])
        #     print("measure state length: ", len(measure_state))

        statistic = self.__statistic_global_state()
        length = len(self.__state_list)
        reward = 0
        for index, state in enumerate(measure_state):
            probability = statistic[str(state)] / length
            reward += probability * measure_reward[index]
        return reward

    def __get_measure_state_reward(self):
        measure_state = []
        measure_reward = []
        for index, state in enumerate(self.__state_list):
            if not self.__is_in_measure_state(measure_state, state):
                measure_state.append(state)
                measure_reward.append(self.__reward_list[index])
        return copy.deepcopy(measure_state), copy.deepcopy(measure_reward)

    @staticmethod
    def __is_in_measure_state(measure_state, state):
        for s in measure_state:
            if hash(str(s)) == hash(str(state)):
                return True
        return False

    def __statistic_global_state(self):
        statistic = {}
        for state in self.__state_list:
            if str(state) not in statistic.keys():
                statistic[str(state)] = 1
            else:
                statistic[str(state)] = statistic[str(state)] + 1
        return copy.deepcopy(statistic)

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

    @property
    def episodes_reward(self):
        return self.__episodes_reward


class AllOffloadAgent:

    def __init__(self):
        pass

    @staticmethod
    def select_action():
        return 1


class AllLocalAgent:

    def __init__(self):
        pass

    @staticmethod
    def select_action():
        return 0


class RandomAgent:
    def __init__(self):
        pass

    @staticmethod
    def select_action():
        return np.random.randint(0, 2)


if __name__ == '__main__':
    # prob = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob = [0]
    from myrun import get_mid_load_prob

    for item in prob:
        p = get_mid_load_prob(item)
        print("带宽概率为：", p)
        env = ECMA(prob=p)
        policy = Policy(env, "random")
        policy.run(60000)
        policy.cal_max_expectation()
