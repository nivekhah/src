from envs.ec.ec_env import ECMA
import numpy as np


class Policy:

    def __init__(self, env: ECMA, policy: str):
        self.__env = env
        self.__episodes_reward = []
        self.__agents = []
        self.__n_agents = env.n_agents
        self.__policy = policy

        self.gen_agent()

    def run(self, t_max):
        episodes = int(t_max / self.__env.MAX_STEPS)
        for i in range(episodes):
            self.__env.reset()
            episode_reward = 0
            for j in range(self.__env.MAX_STEPS):
                self.__env.get_state()
                self.__env.get_obs()

                actions = []
                for agent in self.__agents:
                    action = agent.select_action()
                    actions.append(action)
                reward, done, _ = self.__env.step(actions)
                episode_reward += reward
                print(reward)
            self.__episodes_reward.append(episode_reward)

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
    env = ECMA()

    # ------random---------------
    policy1 = Policy(env, "random")
    policy1.run(6000)
    rewards1 = policy1.episodes_reward

    # -------all_local-------------
    policy2 = Policy(env, "all_local")
    policy2.run(6000)
    rewards2 = policy2.episodes_reward

    # -------all_offload-----------
    policy3 = Policy(env, "all_offload")
    policy3.run(6000)
    rewards3 = policy3.episodes_reward

    import matplotlib.pyplot as plt
    x = [i for i in range(len(policy1.episodes_reward))]
    plt.plot(x, rewards1, label="random", marker="*")
    plt.plot(x, rewards2, label="all_local", marker=">")
    plt.plot(x, rewards3, label="all_offload", marker="o")
    plt.title("episode reward of different policy")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.show()
