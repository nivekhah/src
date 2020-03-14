from src.envs.ec.topology import Topology
import numpy as np
var_vector = [6, 2, 5, 2, 4]
topo = Topology(var_vector)

config = {
}
assert isinstance(config,dict)
config["n_agents"] = 4
config["cl"] = 1
config["cc"] = 10
config["sum_d"] = 10
config["n_actions"] = 2
config["observation_size"] = 2
# config["MAX_STEPS"] = 20
config["MAX_STEPS"] = 20
# config["task_proportion"] = topo.proportion
config["task_proportion"] = [0.25,0.25,0.25,0.25]
config["bandwidth"] = [2,1,0.1]
config["prob"] = [0.2,0.7,1]
