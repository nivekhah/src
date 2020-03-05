from matplotlib import pyplot as plt
import numpy as np
def print_data():
    import json
    filename = "/home/zzw/文档/pymarl/results/sacred/10/info.json"
    file = open(filename, "r")
    data = json.load(file)["return_mean"]
    print(data)

reward = []

def plot_reward():
    reward = np.loadtxt("/home/zzw/文档/pymarl/reward.txt")
    _max = []
    _mean = []
    _min = []
    for i in range(5000):
        temp = reward[20*i:20*(i+1)]
        _max.append(max(temp))
        _min.append(min(temp))
        _mean.append(np.mean(temp))
    plt.plot(list(range(5000))[0:-1:10],_min[0:-1:10], marker="o",color="red", label="min")
    plt.plot(list(range(5000))[0:-1:10],_max[0:-1:10], marker=".",color="black", label="max")
    plt.plot(list(range(5000))[0:-1:10],_mean[0:-1:10], marker="^",color="blue", label="mean")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_reward()