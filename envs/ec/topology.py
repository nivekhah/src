import numpy as np
from numpy.linalg import matrix_rank as rank
class Topology:
    #定义topo
    def __init__(self,matrix,node_matrix,monitor_vector):
        """
        :param matrix 测量矩阵
        [[1,0,1,1,0],
        [1,0,1,0,1],
        [0,1,1,1,0],
        [0,1,1,0,1]]
        :param node_matrix
        [[1,0,0,0,1,0],
        [1,0,0,0,0,1],
        [0,1,0,0,1,0],
        [0,1,0,0,0,1]]
        :param monitor_vector 监控节点
        [1,1,0,0,1,1]
        """
        self.matrix = matrix
        self.node_matrix = node_matrix
        self.monitor_vector = monitor_vector
        self.proportion = self.get_proportion()


    def get_extend_matrix(self):
        """
        获取扩展测量矩阵,返回扩展的测量矩阵
        :return:
        """
        pass

    def matrix_reduction(self,hatR,hatY):
        """
        对测量矩阵规约，返回规约的矩阵和对应Y
        :return:
        """
        dotR = []
        dotY = np.empty(shape=[0, 1])
        C = np.empty(shape=[0, 1])
        while rank(dotR) != rank(hatR):
            rank_dotR = rank(dotR)
            if rank_dotR == 0:
                dotR = np.empty(shape=[0, hatR.shape[1]])
            # 取出最小的行和
            if C.shape[0] == 0:
                index = list(range(hatR.shape[0]))
            else:
                index = list(set(range(hatR.shape[0])).difference(set(C.flatten())))
            temp_hatR = hatR[index]
            k = index[np.random.choice(np.where(np.sum(temp_hatR, 1) == np.min(np.sum(temp_hatR, 1)))[0])]
            if rank(np.row_stack((dotR, hatR[k]))) > rank_dotR:
                dotR = np.row_stack((dotR, hatR[k]))
                dotY = np.row_stack((dotY, hatY[k]))
            C = np.row_stack((C, k))
        print(dotR, dotY)
        return dotR, dotY

    def gen_delay(self):
        """
        生成链路时延,生成路径时延，返回测量Y的值
        :return:
        """
        pass

    def cal_phi(self):
        """
        计算A_i,计算\phi_i,
        :return:
        """

    def cal_proportions(self):
        """
        计算H，返回监测节点的比例
        :return:
        """

    def get_proportion(self):
        pass

