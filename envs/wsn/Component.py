#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'MEC_Q-learning'
@author  : '张宗旺'
@file    : 'Component'.py
@ide     : 'PyCharm'
@time    : '2020'-'01'-'02' '18':'30':'16'
@contact : zongwang.zhang@outlook.com
'''
from abc import ABCMeta, abstractmethod
from envs.wsn.Configuration import config
import numpy as np
import copy


class Component(metaclass=ABCMeta):
    cache = 0
    id = "c"
    @abstractmethod
    def receive_data(self, data):
        pass

    @abstractmethod
    def reset(self):
        pass


class BaseStation(Component):
    id = "b"

    def receive_data(self, data):
        self.cache += data

    def reset(self):
        self.cache = 0


class Satellite(Component):
    id = "s"

    def receive_data(self, data):
        self.cache += data

    def reset(self):
        self.cache = 0


class Sensor(Component):
    # ==============================================#
    # 目前没有考虑传输时候没有传输完的情况          #
    # ==============================================#
    MAX_CACHE = config.get("MAX_CACHE")

    def __init__(self, id):
        self.id = id
        self.connection = config.get("connections")[id]
        self.receive = 0
        self.send = 0
        self.peer_sensor_ids = []
        for peer_id in self.connection:
            if type(peer_id) != str:
                self.peer_sensor_ids.append(peer_id)
        self.state = np.zeros(3)
        self.sample_rate = config.get("sample_rate")
        self.decision_interval = config.get("decision_interval")

    def send_data(self, receiver, data):
        # print("sensor"+str(self.id)+"向"+str(receiver.id)+"发送了"+str(data))
        self.send += data
        assert isinstance(receiver, Component)
        if type(receiver) == Sensor:
            loss_rate = config.get("loss_rate")[0]
            receiver.receive_data(data * (1 - loss_rate))
        elif type(receiver) == BaseStation:
            loss_rate = config.get("loss_rate")[1]
            receiver.receive_data(data * (1 - loss_rate))
        elif type(receiver) == Satellite:
            loss_rate = config.get("loss_rate")[2]
            receiver.receive_data(data * (1 - loss_rate))

    def receive_data(self, data):
        self.receive += data

    def do_action(self, action, components):
        # corresponding_id = int(action / (self.MAX_CACHE+1))
        # corresponding_id = int(action / 2)
        corresponding_id = action
        component_id = self.connection[corresponding_id]
        for component in components:
            assert isinstance(component, Component)
            if component_id == component.id:
                # data = action % (self.MAX_CACHE+1)
                # data = self.cache * (action % 2)
                data = self.cache
                if data > self.cache:
                    self.send_data(component, self.cache)
                    return False
                else:
                    self.send_data(component, data)
                    return True

    def do_settlement(self):
        self.cache = self.cache + self.receive - self.send
        if self.cache > self.MAX_CACHE:
            self.cache = self.MAX_CACHE
        self.send = 0
        self.receive = 0

    def reset(self):
        self.cache = 0
        self.send = 0
        self.receive = 0
        if BaseStation.id in self.connection:
            self.state[0] = -1
        else:
            self.state[0] = 0
        self.state[1] = 0
        self.state[1] = 0

    def get_observation(self, peer_sensors):
        assert isinstance(peer_sensors, list)
        if BaseStation.id in self.connection:
            self.state[0] = -1
            for peer_id in self.peer_sensor_ids:
                for sensor in peer_sensors:
                    if peer_id == sensor.id:
                        assert isinstance(sensor, Sensor)
                        # self.state[1] = sensor.MAX_CACHE - sensor.cache
                        self.state[1] = sensor.cache
        else:
            for peer_id in self.peer_sensor_ids:
                for sensor in peer_sensors:
                    if peer_id == sensor.id:
                        assert isinstance(sensor, Sensor)
                        # self.state[self.peer_sensor_ids.index(peer_id)] = sensor.MAX_CACHE - sensor.cache
                        self.state[self.peer_sensor_ids.index(peer_id)] = sensor.cache
        self.state[-1] = self.cache
        # return str(self.state)
        return copy.deepcopy(self.state)

    def sample_data(self):
        self.cache = self.decision_interval*self.sample_rate+self.cache
        if self.cache > self.MAX_CACHE:
            self.cache = self.MAX_CACHE

