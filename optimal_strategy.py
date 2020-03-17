import pandas as pd
import os
from envs.ec.modify_yaml import ModifyYAML


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


def is_optimal(state, action, reward, args):
    actions = []
    times = []
    for i in range(int(len(state) / 2)):
        obs = [state[i * 2], state[i * 2 + 1]]  # 依次解析得到每一个 agent 的 observation，并且计算理论上的最优策略以及对应的处理时长
        a, t = agent_optimal(obs, args)
        actions.append(a)
        times.append(t)
    rewards = args["env_args"]["sum_d"] / max(times)
    if actions != action:
        print(state, "\t", action, "\t", actions, "\t", reward, "\t", rewards)


if __name__ == '__main__':
    state_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "rl_state.csv")
    data = pd.read_csv(state_path)
    state = data["state"]
    action = data["action"]
    reward = data["reward"]
    # print(reward)
    # print(state)

    ec_modify = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "envs", "ec.yaml"))

    for index, item in enumerate(state):
        s = list(map(float, state[index].split("[")[1].split("]")[0].split(",")))
        a = list(map(int, action[index].split("[")[1].split("]")[0].split(",")))
        r = float(reward[index])
        # r = list(map(float, action[index].split("[")[1].split("]")[0].split(",")))
        is_optimal(s, a, r, ec_modify.data)
