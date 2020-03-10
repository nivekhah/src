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
            self.__episodes_reward = []

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
    policy = Policy(env, "random")
    policy.run(60)
