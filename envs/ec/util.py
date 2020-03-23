from envs.ec.ec_env import ECMA
from envs.ec.policy import Policy


def statistic_action_for_agent(policy: Policy):
    """
    统计 policy 策略下，每一个 agent 对应的每一种 action 出现的次数

    ---------
    返回结果样例：[{action_1: number_1}, {action_2: number_2}, {action_3: number_3}, {action_4: number_4}]
    第 i 个元素对应着第 i 个 Edge Server.

    :param policy: 策略对象
    :return:
    """
    assert len(policy.total_action) != 0
    statistic = [{}, {}, {}, {}]
    for action in policy.total_action:
        for i in range(len(action)):
            if action[i] in statistic[i].keys():
                statistic[i][action[i]] = statistic[i][action[i]] + 1
            else:
                statistic[i][action[i]] = 1
    return statistic


def statistic_reward(policy):
    """
    统计 policy 策略下，每一种 reward 出现的次数

    ---------
    返回结果样例：{reward_1: number_1, reward_2: number_2, reward_3: number_3, ...}

    :param policy:
    :return:
    """
    assert len(policy.total_reward) != 0
    sta_reward = {}
    for reward in policy.total_reward:
        if reward in sta_reward.keys():
            sta_reward[reward] = sta_reward[reward] + 1
        else:
            sta_reward[reward] = 1
    return sta_reward


def statistic_all_item(policy):
    """
    统计 policy 策略下所有出现的 state-action-reward 组合的情况，以及其出现的数量

    ------------
    返回的结果样例：
    {state_1: {action_1: {reward_1: number_1}
                         {reward_2: number_2}
                         ...}
              {action_2: {reward_3: number_3}
                         {reward_4: number_4}
                         ...}
              ...
    ...}


    :param policy:
    :return:
    """
    statistic = {}
    total_state = policy.total_state
    total_action = policy.total_action
    total_reward = policy.total_reward
    for index, state in enumerate(total_state):
        current_state = str(state)
        current_action = str(total_action[index])
        current_reward = str(total_reward[index])
        if current_state not in statistic.keys():
            statistic[current_state] = {current_action: {current_reward: 1}}
        else:
            if current_action not in statistic[current_state].keys():
                statistic[current_state][current_action] = {current_reward: 1}
            else:
                if current_reward not in statistic[current_state][current_action].keys():
                    statistic[current_state][current_action][current_reward] = 1
                else:
                    statistic[current_state][current_action][current_reward] = \
                        statistic[current_state][current_action][current_reward] + 1
    return statistic


def plot_random_reward_light_load():
    """
    画出 random 策略下，期望 reward 随着轻载概率的变化而变化的图像
    :return:
    """
    p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    prob = [[0, 0.5, 1],
            [0.1, 0.55, 1],
            [0.2, 0.6, 1],
            [0.3, 0.65, 1],
            [0.4, 0.7, 1],
            [0.5, 0.75, 1],
            [0.6, 0.8, 1],
            [0.7, 0.85, 1],
            [0.8, 0.9, 1],
            [0.9, 0.95, 1],
            [1.0, 1.0, 1]]

    reward_list = []
    for item in prob:
        env_temp = ECMA(prob=item)
        temp_alg = Policy(env_temp, "random")
        temp_alg.run(200)
        from envs.ec.expected_reward import ExpectedReward

        expected_reward = ExpectedReward(temp_alg.total_state, temp_alg.total_reward).get_expected_reward()
        reward_list.append(expected_reward)
    # print(reward_list)

    from matplotlib import pyplot as plt

    plt.plot(p, reward_list)
    plt.xticks(p)
    plt.xlabel("probability of light load")
    plt.ylabel(r"expected reward $\bar{r}$")
    plt.show()


if __name__ == '__main__':
    plot_random_reward_light_load()
