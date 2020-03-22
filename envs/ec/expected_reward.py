"""
ExpectedReward 类主要用于根据记录的所有的 state, reward 数据来计算 reward 的期望值
依据的算法为： MAC(Maximum Expected Amount of End-to-End Travel Time Collection)
"""

import copy


class ExpectedReward:
    def __init__(self, total_state: list, total_reward: list):
        assert len(total_state) == len(total_reward)
        self.__total_state = copy.deepcopy(total_state)
        self.__total_reward = copy.deepcopy(total_reward)

    def get_expected_reward(self):
        """
        统计 self.__total_state 和 self.__total_reward 组成的 state-reward 的对数，以及每一对出现的次数，并且依据统计的信息计算
        reward 的加权平均值
        :return:
        """
        statistics = self.__statistic()
        expected_reward = 0
        len_state = len(self.__total_state)
        for key, value in statistics.items():
            reward = value[0]
            count = value[1]
            prob = count / len_state
            expected_reward += reward * prob
        return expected_reward

    def __statistic(self):
        """
        统计 self__total_state 和 self_total_reward 中总共出现了多少种 state-reward 对，以及每一对出现了多少次.
        statistics 统计的样例如下：
        {
            "[state, reward]": [reward, count]
        }
        :return: 统计结果 dict
        """
        statistics = {}
        for index, state in enumerate(self.__total_state):
            reward = self.__total_reward[index]
            key = str([state, reward])
            if key in statistics.keys():
                value = statistics[key]
                reward = value[0]
                count = value[1]
                statistics[key] = [reward, count + 1]
            else:
                statistics[key] = [reward, 1]
        return statistics
