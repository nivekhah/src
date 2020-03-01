import numpy as np
from src.envs.ec.config import config
import copy


class EdgeServer:
    def __init__(self, id: int, cl):
        """
        :param id: edge server 的编号
        :param cl: edge server 的计算速度
        """
        self.id = id
        self.d = -1  # 初始化任务量为 0
        self.cl = cl  # 本地计算速度
        self.b = -1  # 带宽

    def do_action(self, action, tcc):
        time = 0
        if action == 0:
            time = self.do_local()
        elif action == 1:
            time = self.offload_tcc(tcc)
        self.d = 0
        return time

    def get_obs(self):
        """
        获取 edge server 此刻的观测值

        edge server 此刻的观测值由两部分构成：
        1. 此刻分配给 edge server 的任务；
        2. 此刻 edge server 的可用带宽。

        :return: edge server 此刻的观测值
        """
        return np.array([self.d, self.b])

    def do_local(self):
        """
        当 edge server 决定本地处理计算任务时，其处理任务的时间由两部分构成：
        1. edge server 计算任务的时间：为任务大小除以本地处理速度
            $$
            tc = \frac{d}{cl}
            $$
        2. edge server 传输任务的时间：0 （不需要传送）
            $$
            tb = 0
            $$

        :return: edge server 处理任务的时间
        """
        tb = 0  # 传输时间
        tc = self.d / self.cl  # 本地处理时间
        time = tb + tc
        return time

    def offload_tcc(self, tcc):
        """
        当 edge server 决定将任务 offload 到 TCC 进行处理时，处理任务的时间由两部分组成：
        1. 传输到 TCC 所需要的传输时间：任务大小除以带宽
            $$
            tb = \frac{d}{b}
            $$
        2. TCC 计算任务的时间：任务大小除以处理速度
            $$
            tc = \frac{d}{cc}
            $$

        :param tcc:
        :return:
        """
        assert isinstance(tcc, TCC)
        tb = self.d / self.b
        tc = tcc.do_task(self.d)
        time = tb + tc
        return time

    def get_available_bandwidth(self):
        bandwidth = config.get("bandwidth")
        # prob = config.get("prob")
        # p = np.random.uniform(0,1)
        # copy_prob = copy.deepcopy(prob)
        #
        # copy_prob.append(p)
        # copy_prob.sort()
        # return bandwidth[copy_prob.index(p)]
        a = [1,1,0]
        return bandwidth[a[self.id]]

    def reset(self, d):
        self.next_step(d)

    def next_step(self, d):
        self.d = d
        self.b = self.get_available_bandwidth()


class TCC:
    def __init__(self, cc):
        """
        :param cc: TCC 的计算速度
        """
        self.id_ = "tcc"
        self.cc = cc  # tcc的处理速度

    def do_task(self, d):
        """
        TCC 计算任务 d 所花费的时间为任务大小除以 TCC 处理任务的速度，即：

        $$
        tc = \frac{d}{cc}
        $$

        :param d: 任务 d
        :return: TCC 计算任务 d 所花费的时间
        """
        tc = d / self.cc
        return tc

    def reset(self):
        pass
