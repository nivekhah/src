config = {
}
assert isinstance(config,dict)
config["n_agents"] = 3
config["cl"] = 1
config["cc"] = 10
config["sum_d"] = 10
config["n_actions"] = 2
config["observation_size"] = 2
config["MAX_STEPS"] = 20
config["task_proportion"] = [0.2,0.5,0.3]
config["bandwidth"] = [2,1,0.1]
config["prob"] = [0.2,0.7,1]
