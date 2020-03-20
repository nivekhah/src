from envs.ec.modify_yaml import ModifyYAML
import os
import numpy as np


file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "test.txt")

state = np.loadtxt(file_path).tolist()

action = [[1, 1, 1, 0],
          [1, 1, 1, 1],
          [1, 1, 0, 0],
          [1, 1, 0, 1],
          [1, 0, 1, 0],
          [1, 0, 1, 1],
          [1, 0, 0, 0],
          [1, 0, 0, 1],
          [0, 1, 1, 0],
          [0, 1, 1, 1],
          [0, 1, 0, 0],
          [0, 1, 0, 1],
          [0, 0, 1, 0],
          [0, 0, 1, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 1]]


def get_local_time(obs):
    cl = 2.6
    time = obs[1] / cl
    return time


def get_tcc_time(obs):
    cc = 104
    tb = obs[1] / obs[0]
    tc = obs[1] / cc
    return tb + tc


def get_reward(s, a):
    times = []
    for i in range(len(a)):
        # print(i)
        obs = [s[i*2], s[i*2 + 1]]
        if a[i] == 0:
            times.append(get_local_time(obs))
        else:
            times.append(get_tcc_time(obs))
    print("state: ", s, "action: ", a, "time:", times)
    sum_d = 644.84
    return sum_d / max(times)


def to_string(data):
    res = ""
    for i in data:
        res += str(i) + "_"
    return res


def main():
    # res = list()
    # temp = list()
    # temp.append([0])
    # for a in action:
    #     temp.append(to_string(a))
    # temp.append("max")
    # res.append(temp)
    # print(temp)

    # res = list()
    # for a in action:
    #     temp = list()
    #     for s in state:
    #         temp.append(get_reward(s, a))
    #     res.append(temp)
    #
    # for s in state:
    #     x = []
    #     r_t = []
    #     x.append(to_string(s))
    #     for a in action:
    #         r = get_reward(s, a)
    #         x.append(r)
    #         r_t.append(r)
    #     x.append(max(r_t))
    #     print(x)
    #     res.append(x)

    s = state[26]
    for a in action:
        get_reward(s, a)

    # print(res)
    # np.savetxt("test.txt", res)


if __name__ == '__main__':
    main()
