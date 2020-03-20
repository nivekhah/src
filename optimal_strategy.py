import pandas as pd
import os
from envs.ec.modify_yaml import ModifyYAML
import matplotlib.pyplot as plt
import numpy as np


def do_local(task, args):
    """
    :param task 任务大小
    :param args 环境参数
    """
    cl = args["env_args"]["cl"]
    time = float(task) / cl
    return time


def do_offload(bandwidth, task, args):
    """
    :param bandwidth 链路带宽
    :param task 任务大小
    :param args 环境参数
    :return 返回 TCC 处理任务的时长
    """
    cc = args["env_args"]["cc"]
    time_trans = task / bandwidth
    time_deal = task / cc
    return time_trans + time_deal


def agent_optimal(obs, args):
    """
    获取
    :param obs agent 对应的 observation
    :return 返回最优动作，以及最优动作下对应的任务处理时长
    """
    time_local = do_local(obs[1], args)
    time_offload = do_offload(obs[0], obs[1], args)
    if time_local > time_offload:
        return 1, time_offload
    elif time_local < time_offload:
        return 0, time_local


def get_optimal(state, args):
    actions = []
    times = []
    for i in range(int(len(state) / 2)):
        obs = [state[i * 2], state[i * 2 + 1]]  # 依次解析得到每一个 agent 的 observation，并且计算理论上的最优策略以及对应的处理时长
        a, t = agent_optimal(obs, args)
        actions.append(a)
        times.append(t)

    rewards = args["env_args"]["sum_d"] / max(times)
    # if actions != action:
    #     print(state, "\t", action, "\t", actions, "\t", reward, "\t", rewards)
    return rewards


def process_train_state():
    state_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "train_state.txt")
    reward_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "train_reward.txt")
    op_reward_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "optimal_reward.txt")
    error_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "error.txt")
    state = np.loadtxt(state_path).tolist()
    # np.savetxt(state_path, state, fmt="%.2f")
    reward = np.loadtxt(reward_path).tolist()
    ec_modify = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "envs", "ec.yaml"))
    optimal_reward = []
    error_list = []
    for index, item in enumerate(state):
        o_r = get_optimal(item, ec_modify.data)
        optimal_reward.append(o_r)
        error = (round(o_r, 5) - round(reward[index], 5)) / round(o_r, 5)
        error_list.append(error)
        if error < 0:
            print(error)
    np.savetxt(op_reward_path, optimal_reward)
    np.savetxt(error_path, error_list)
    print(error_list)
    # print(optimal_reward)
    # print(reward)
    # res = np.subtract(optimal_reward, reward)
    # print(res)


if __name__ == '__main__':
    process_train_state()
    # state_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "rl_state.csv")
    # data = pd.read_csv(state_path)
    # state_ = data["state"]
    # action_ = data["action"]
    # reward_ = data["reward"]
    # print(reward)
    # print(state)

    # ec_modify = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "envs", "ec.yaml"))
    #
    # hashcode = []
    # state_statistic = []
    # reward_statistic = []
    # opt_reward_statistic = []
    #
    # count = 0
    #
    # for index, item in enumerate(state_):
    #     s = list(map(float, state_[index].split("[")[1].split("]")[0].split(",")))
    #     a = list(map(int, action_[index].split("[")[1].split("]")[0].split(",")))
    #     r = float(reward_[index])

        # code = hash(str(s))
        # if code not in hashcode:
        #     hashcode.append(code)
        #     state_statistic.append(str(s))
        #     reward_statistic.append(r)
        #     temp_reward = get_optimal(s, a, r, ec_modify.data)
        #     opt_reward_statistic.append(temp_reward)

        # reward = get_optimal(s, a, r, ec_modify.data)
        #
        # if reward == r:
        #     count += 1

    # print(count/6000)
    # print(len(state_statistic))
    # x = [i for i in range(len(state_statistic))]
    # plt.plot(x, reward_statistic, label="QMIX")
    # plt.plot(x, opt_reward_statistic, label="Optimal")
    # # plt.xticks(state_statistic)
    # plt.xlabel("state")
    # plt.ylabel("reward")
    # plt.legend()
    # plt.show()
