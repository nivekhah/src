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
import json

json_file = open("src/envs/wsn/config.json","r",encoding="utf-8")
config = json.load(json_file)
assert isinstance(config, dict)
# print(type(config.get("decision_interval")))
