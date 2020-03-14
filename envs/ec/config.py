from src.envs.ec.topology import Topology
import numpy as np

measure_matrix = np.array([[1, 0, 1, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 0, 1]])
node_matrix = np.array([[1, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 1]])
monitor_vector = np.array([1, 1, 0, 0, 1, 1])
var_vector = [6, 2, 5, 2, 4]
topo = Topology(measure_matrix, node_matrix, monitor_vector, var_vector)

config = {
}
assert isinstance(config, dict)
config["n_agents"] = 4
config["cl"] = 1
config["cc"] = 4
config["sum_d"] = 10
config["n_actions"] = 2
config["observation_size"] = 2
# config["MAX_STEPS"] = 20
config["MAX_STEPS"] = 20
# config["task_proportion"] = topo.proportion
config["task_proportion"] = [0.4, 0.1, 0.2, 0.3]
# config["task_proportion"] = [0.25, 0.25, 0.25, 0.25]
config["bandwidth"] = [2, 1, 0.1]
# config["prob"] = [0.2, 0.7, 1]

# config["prob"] = [0.1, 0.55, 1]
# config["prob"] = [0.2, 0.6, 1]
# config["prob"] = [0.3, 0.65, 1]
# config["prob"] = [0.4, 0.7, 1]
# config["prob"] = [0.5, 0.75, 1]
# config["prob"] = [0.6, 0.8, 1]
# config["prob"] = [0.7, 0.85, 1]
# config["prob"] = [0.8, 0.9, 1]
# config["prob"] = [0.9, 0.95, 1]
config["prob"] = [1, 1, 1]
