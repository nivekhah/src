from envs.ec.modify_yaml import ModifyYAML
from envs.ec.policy import Policy
from envs.ec.ec_env import ECMA

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
import copy


def cc_cl_scale(modify):
    """
    此方法用于训练 cc/cl 为不同值时的模型
    :param modify:
    :return:
    """
    write_train_default_config()

    data = modify.data
    cl_cl = data["cc_cl_scale"]
    for scale in cl_cl:
        data["env_args"]["cc"] = scale
        modify.dump()

        my_main()


def light_load_prob(modify):
    """
    此方法用于训练链路轻载概率不同，而中载和重载的概率相同时的模型
    :param modify:
    :return:
    """
    write_train_default_config()

    data = modify.data
    probs = data["light_load_prob"]
    for p in probs:
        prob = list()
        prob.append(p)
        prob.append(round((p + (1 - p) / 2), 2))
        prob.append(1)
        data["env_args"]["prob"] = prob
        modify.dump()

        my_main()


def mid_load_prob(modify):
    """
    此方法用于训练链路中载概率不同，而轻载和重载的概率相同时的模型
    :param modify:
    :return:
    """
    write_train_default_config()

    data = modify.data
    probs = data["mid_load_prob"]
    for p in probs:
        prob = list()
        prob.append(round((1 - p) / 2, 2))
        prob.append(round(((1 - p) / 2) + p, 2))
        prob.append(1)
        print("带宽分配概率： ", prob)
        data["env_args"]["prob"] = prob
        modify.dump()

        my_main()


def weight_load_prob(modify):
    """
    此方法用于训练链路重载概率不同，而轻载和中载的概率相同时的模型
    :param modify:
    :return:
    """
    write_train_default_config()

    data = modify.data
    probs = data["weight_load_prob"]
    for p in probs:
        prob = list()
        prob.append(round((1 - p) / 2, 2))
        prob.append(round(1 - p, 2))
        prob.append(1)
        print("带宽分配概率： ", prob)
        data["env_args"]["prob"] = prob
        modify.dump()

        my_main()


def gen_data_cc_cl(modify):
    """
    此方法用于生成不同算法对应的最大期望任务数据

    :param modify:
    :return:
    """
    data = modify.data
    checkpoint_path = data["cc_cl_checkpoint_path"]
    cc_cl = data["cc_cl_scale"]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for index, scale in enumerate(cc_cl):
        data["env_args"]["cc"] = scale
        modify.dump()

        write_gen_default_config(checkpoint_path[index])

        env = gen_env(data["env_args"])

        my_main()

        gen_t_max = data["gen_t_max"]
        run_policy(env, "random", gen_t_max, "cc_cl")
        run_policy(env, "all_local", gen_t_max, "cc_cl")
        run_policy(env, "all_offload", gen_t_max, "cc_cl")


def gen_data_light_load(modify):
    """
    生成轻载服从不同概率时，不同算法
    :param modify:
    :return:
    """

    data = modify.data

    prob = data["light_load_prob"]  # 为中载设置的所有可能的概率[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    checkpoint_path = data["light_load_checkpoint_path"]
    for index, p in enumerate(prob):
        data["env_args"]["prob"] = get_light_load_prob(p)
        modify.dump()

        write_gen_default_config(checkpoint_path[index])

        env = gen_env(data["env_args"])

        my_main()

        gen_t_max = data["gen_t_max"]
        run_policy(env, "random", gen_t_max, "light_load")
        run_policy(env, "all_local", gen_t_max, "light_load")
        run_policy(env, "all_offload", gen_t_max, "light_load")


def get_light_load_prob(prob):
    res = list()
    res.append(prob)
    res.append(round((prob + (1 - prob) / 2), 2))
    res.append(1)
    return copy.deepcopy(res)


def gen_data_mid_load(modify):
    # 加载 ec.yaml 的配置项
    data = modify.data

    prob = data["mid_load_prob"]  # 为中载设置的所有可能的概率[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    checkpoint_path = data["mid_load_checkpoint_path"]
    for index, p in enumerate(prob):
        data["env_args"]["prob"] = get_mid_load_prob(p)
        print("带宽概率： ", data["env_args"]["prob"])
        modify.dump()

        write_gen_default_config(checkpoint_path[index])

        env = gen_env(data["env_args"])

        my_main()

        gen_t_max = data["gen_t_max"]
        run_policy(env, "random", gen_t_max, "mid_load")
        run_policy(env, "all_local", gen_t_max, "mid_load")
        run_policy(env, "all_offload", gen_t_max, "mid_load")


def get_mid_load_prob(prob):
    res = list()
    res.append(round((1 - prob) / 2, 2))
    res.append(round(((1 - prob) / 2) + prob, 2))
    res.append(1)
    return copy.deepcopy(res)


def gen_data_weight_load(modify):
    # 加载 ec.yaml 的配置项
    data = modify.data

    prob = data["weight_load_prob"]  # 为中载设置的所有可能的概率[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    checkpoint_path = data["weight_load_checkpoint_path"]
    for index, p in enumerate(prob):
        data["env_args"]["prob"] = get_weight_load_prob(p)
        modify.dump()

        write_gen_default_config(checkpoint_path[index])

        env = gen_env(data["env_args"])

        my_main()

        gen_t_max = data["gen_t_max"]
        run_policy(env, "random", gen_t_max, "weight_load")
        run_policy(env, "all_local", gen_t_max, "weight_load")
        run_policy(env, "all_offload", gen_t_max, "weight_load")


def get_weight_load_prob(prob):
    res = list()
    res.append(round((1 - prob) / 2, 2))
    res.append(round(1 - prob, 2))
    res.append(1)
    return copy.deepcopy(res)


def gen_env(env_args):
    env = ECMA(seed=None,
               max_steps=env_args["max_steps"],
               bandwidth=env_args["bandwidth"],
               cc=env_args["cc"],
               cl=env_args["cl"],
               n_agents=env_args["n_agents"],
               n_actions=env_args["n_actions"],
               observation_size=env_args["observation_size"],
               prob=env_args["prob"],
               sum_d=env_args["sum_d"],
               task_proportion=env_args["task_proportion"])
    return env


def write_train_default_config():
    """
    在每次训练之前，将
    :return:
    """
    default = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "default.yaml"))
    default_data = default.data
    default_data["checkpoint_path"] = ""
    default_data["cal_max_expectation_tasks"] = False
    default.dump()


def write_gen_default_config(checkpoint_path):
    default = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "default.yaml"))
    default_data = default.data
    default_data["checkpoint_path"] = os.path.join(os.path.dirname(__file__), "results", "models", checkpoint_path)
    default_data["cal_max_expectation_tasks"] = True
    default.dump()


def run_policy(env, policy_name, t_max, flag):
    policy = Policy(env, policy_name)
    policy.run(t_max)
    reward = policy.cal_max_expectation()

    file_path = get_data_file_path(policy_name, flag)
    with open(file_path, 'a') as f:
        f.write(str(reward) + "\n")


def get_data_file_path(policy_name, flag):
    """
    获取最大期望任务存储文件的路径
    :param policy_name: 算法名，有四种，依次为： random, all_local, all_offload, rl
    :param flag: 表示针对哪一个配置项进行研究，有四个： cc_cl, light_load, mid_load, weight_load
    :return: 存储数据的相应文件的路径
    """
    return os.path.join(os.path.dirname(__file__), "envs", "ec", "output", policy_name + "_" + flag + ".txt")


def get_ec_config_file_path():
    """
    返回 ec.yaml 配置文件的路径
    :return: 路径
    """
    return os.path.join(os.path.dirname(__file__), "config", "envs", "ec.yaml")


def plot_scale():
    flag = "cc_cl"
    random_mean_reward, all_local_max_expectation, all_offload_max_expectation, rl_max_expectation = get_reward_data(
        flag)
    x = ModifyYAML(get_ec_config_file_path()).data["cc_cl_scale"]
    png_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "cc_cl.png")
    eps_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "cc_cl.eps")
    plot(x, random_mean_reward, all_local_max_expectation, all_offload_max_expectation,
         rl_max_expectation, r"$C_c / C_l$", r"normalized $\bar{\psi}$", png_file_path, eps_file_path)


def plot_light_load():
    flag = "light_load"
    random_mean_reward, all_local_max_expectation, all_offload_max_expectation, rl_max_expectation = get_reward_data(
        flag)

    x = ModifyYAML(get_ec_config_file_path()).data["light_load_prob"]
    png_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "light_load.png")
    eps_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "light_load.eps")
    plot(x, random_mean_reward, all_local_max_expectation, all_offload_max_expectation,
         rl_max_expectation, " probability of light load", r"normalized $\bar{\psi}$", png_file_path, eps_file_path)


def plot_mid_load():
    flag = "mid_load"
    random_mean_reward, all_local_max_expectation, all_offload_max_expectation, rl_max_expectation = get_reward_data(
        flag)

    x = ModifyYAML(get_ec_config_file_path()).data["mid_load_prob"]
    png_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "moderate_load.png")
    eps_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "moderate_load.eps")
    plot(x, random_mean_reward, all_local_max_expectation, all_offload_max_expectation,
         rl_max_expectation, "probability of moderate load", r"normalized $\bar{\psi}$", png_file_path, eps_file_path)


def plot_weight_load():
    flag = "weight_load"
    random_mean_reward, all_local_max_expectation, all_offload_max_expectation, rl_max_expectation = get_reward_data(
        flag)

    x = ModifyYAML(get_ec_config_file_path()).data["light_load_prob"]
    png_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "heavy_load.png")
    eps_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "heavy_load.eps")
    plot(x, random_mean_reward, all_local_max_expectation, all_offload_max_expectation,
         rl_max_expectation, "probability of heavy load", r"normalized $\bar{\psi}$", png_file_path, eps_file_path)


def plot(x, random_mean_reward,
         all_local_max_expectation,
         all_offload_max_expectation,
         rl_max_expectation,
         x_label,
         y_label,
         png_file_path,
         eps_file_path):
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    # rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    print(x,rl_max_expectation.tolist(),random_mean_reward.tolist(),all_local_max_expectation.tolist(),all_offload_max_expectation.tolist())
    print(x_label,y_label)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(x, fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig(png_file_path, format="png")
    plt.savefig(eps_file_path, format="eps")
    plt.show()


def plot_reward(reward_path):
    import json

    with open(reward_path, 'r') as f:
        reward = np.array(list(json.load(f)["return_mean"]))
        max_reward = np.max(reward)
        normal_reward = np.divide(reward, max_reward)

    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    # x = [i for i in range(len(normal_reward))]
    plt.figure(figsize=(8, 6))
    plt.plot(normal_reward, 'r')
    plt.xlabel(r"time step ($10^2$)", fontsize=12)
    plt.ylabel("normalized $r$", fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xticks(x, fontsize=10)
    plt.xticks(fontsize=12)
    plt.grid()
    png_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "reward.png")
    eps_file_path = os.path.join(os.path.dirname(__file__), "envs", "ec", "output", "reward.eps")
    plt.savefig(png_file_path, format="png")
    plt.savefig(eps_file_path, format="eps")
    plt.show()


def get_reward_data(flag):
    """
    在文件中读取四种不同的算法对应的
    :param flag:
    :return:
    """
    random_mean_reward = np.loadtxt(get_data_file_path("random", flag))
    all_local_max_expectation = np.loadtxt(get_data_file_path("all_local", flag))
    all_offload_max_expectation = np.loadtxt(get_data_file_path("all_offload", flag))
    rl_max_expectation = np.loadtxt(get_data_file_path("rl", flag))
    reward = [np.max(random_mean_reward), np.max(all_local_max_expectation),
              np.max(all_offload_max_expectation), np.max(rl_max_expectation)]
    max_reward = np.max(reward)
    random_mean_reward = np.divide(random_mean_reward, max_reward)
    all_local_max_expectation = np.divide(all_local_max_expectation, max_reward)
    all_offload_max_expectation = np.divide(all_offload_max_expectation, max_reward)
    rl_max_expectation = np.divide(rl_max_expectation, max_reward)

    return random_mean_reward, all_local_max_expectation, all_offload_max_expectation, rl_max_expectation


def my_main():
    """
    主函数，系统调用 main.py
    :return:
    """
    os.system("python3 /home/csyi/pymarl/src/main.py --env-config=ec --config=qmix")


if __name__ == '__main__':
    ec_modify = ModifyYAML(os.path.join(os.path.dirname(__file__), "config", "envs", "ec.yaml"))

    # 当 train_cc_cl_scale 为 true 时， 表示此时会修改 cc 和 cl 参数，并且训练模型
    # 此时必须设置 cc_cl_scale 参数，而且 cc/cl 的比例会依次为 cc_cl_scale 中的值
    if ec_modify.data["train_cc_cl_scale"] is True:
        cc_cl_scale(ec_modify)

    # 当 train_light_load_prob 为 true 时，表示此时会修改 prob 参数，并且训练模型
    # 此时必须设置 light_load_prob 参数， 从 light_load_prob 中依次取值来当做轻载的概率
    if ec_modify.data["train_light_load_prob"] is True:
        light_load_prob(ec_modify)

    # 当 train_mid_load_prob 为 true 时，表示此时会修改 prob 参数，并且训练模型
    # 此时必须设置 mid_load_prob 参数， 从 mid_load_prob 中依次取值来当做中载的概率
    if ec_modify.data["train_mid_load_prob"] is True:
        mid_load_prob(ec_modify)

    # 当 train_weight_load_prob 为 true 时，表示此时会修改 prob 参数，并且训练模型
    # 此时必须设置 weight_load_prob 参数， 从 weight_load_prob 中依次取值来当做中载的概率
    if ec_modify.data["train_weight_load_prob"] is True:
        weight_load_prob(ec_modify)

    # 当 gen_data_cc_cl 为 true 时，会根据之前训练的模型生成最大期望任务数据
    # 此时需要设置相应的模型路径参数：cc_cl_checkpoint_path
    if ec_modify.data["gen_data_cc_cl"] is True:
        gen_data_cc_cl(ec_modify)

    # 当 gen_data_cc_cl 为 true 时，会根据之前训练的模型生成最大期望任务数据
    # 此时需要设置相应的模型路径参数：light_load_checkpoint_path
    if ec_modify.data["gen_data_light_load"] is True:
        gen_data_light_load(ec_modify)

    # 当 gen_data_cc_cl 为 true 时，会根据之前训练的模型生成最大期望任务数据
    # 此时需要设置相应的模型路径参数：mid_load_checkpoint_path
    if ec_modify.data["gen_data_mid_load"] is True:
        gen_data_mid_load(ec_modify)

    # 当 gen_data_cc_cl 为 true 时，会根据之前训练的模型生成最大期望任务数据
    # 此时需要设置相应的模型路径参数：weight_load_checkpoint_path
    if ec_modify.data["gen_data_weight_load"] is True:
        gen_data_weight_load(ec_modify)

    if ec_modify.data["plot_cc_cl_scale"] is True:
        plot_scale()

    if ec_modify.data["plot_light_load"] is True:
        plot_light_load()

    if ec_modify.data["plot_mid_load"] is True:
        plot_mid_load()

    if ec_modify.data["plot_weight_load"] is True:
        plot_weight_load()

    if ec_modify.data["plot_reward"] is True:
        plot_reward(ec_modify.data["reward_path"])
