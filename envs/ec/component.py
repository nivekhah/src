import numpy as np
from src.envs.ec.config import config
import copy
class EdgeServer:
    def __init__(self,id:int,cl):
        self.id = id
        self.d = -1  #初始化任务量为0
        self.cl = cl  #本地计算速度
        self.b = -1 #带宽

    def do_action(self,action, tcc):
        time = 0
        if action == 0:
            time = self.do_local()
        elif action == 1:
            time = self.offload_tcc(tcc)
        self.d = 0
        return time

    def get_obs(self):
        return np.array([self.d, self.b])

    def do_local(self):
        tb = 0 #传输时间
        tc = self.d/self.cl #本地处理时间
        time = tb+tc
        return  time


    def offload_tcc(self,tcc):
        assert isinstance(tcc,TCC)
        tb = self.d/self.b
        tc = tcc.do_task(self.d)
        time = tb+tc
        return time

    def get_available_bandwidth(self):
        bandwidth = config.get("bandwidth")
        prob = config.get("prob")
        p = np.random.uniform(0,1)
        copy_prob = copy.deepcopy(prob)

        copy_prob.append(p)
        copy_prob.sort()
        return bandwidth[copy_prob.index(p)]
        # a = [1,1,0]
        # return bandwidth[a[self.id]]





    def reset(self,d):
        self.next_step(d)

    def next_step(self,d):
        self.d = d
        self.b = self.get_available_bandwidth()


class TCC:
    def __init__(self, cc):
        self.id_= "tcc"
        self.cc = cc  ##tcc的处理速度

    def do_task(self, d):
        tc = d/self.cc
        return tc

    def reset(self):
        pass