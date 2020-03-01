#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'MEC_Q-learning'
@author  : '张宗旺'
@file    : 'Configuration'.py
@ide     : 'PyCharm'
@time    : '2020'-'01'-'02' '17':'59':'21'
@contact : zongwang.zhang@outlook.com
'''
config = {
}
assert isinstance(config,dict)
config["connections"] = [["s", 1, "null"], ["s", 2, 0], ["s", "b", 1]]
config["sensor_num"] = 3
config["MAX_CACHE_SIZE"] = [10] * config.get("sensor_num")
config["loss_rate"] = [
    [0.15, 0.05, 0],
    [0.15, 0.05, 0.05],
    [0.15, 0.1, 0.05]
]
config["MAX_STEPS"] = 100
config["decision_interval"] = 2
config["sample_rate"] = 1

config["n_actions"] = 2
config["observation_size"] = 3
config["state_last_action"] = True