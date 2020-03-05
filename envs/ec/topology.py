import numpy as np
from numpy.linalg import matrix_rank as rank


class Topology:
    # 定义topo
    def __init__(self, matrix, node_matrix, monitor_vector, var_vector):
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
        :param var_vector 链路时延方差向量
        [6, 2, 5, 2, 4]
        """
        self.matrix = matrix
        self.node_matrix = node_matrix
        self.monitor_vector = monitor_vector
        self.var_vector = var_vector
        # self.proportion = self.get_proportion()

    def get_extend_matrix(self):
        """
        获取扩展测量矩阵,返回扩展的测量矩阵
        :return:
        """
        pass

    def matrix_reduction(self, hatR, hatY):
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
                index = list(
                    set(range(hatR.shape[0])).difference(set(C.flatten())))
            temp_hatR = hatR[index]
            k = index[np.random.choice(
                np.where(np.sum(temp_hatR, 1) == np.min(np.sum(temp_hatR, 1)))[0])]
            if rank(np.row_stack((dotR, hatR[k]))) > rank_dotR:
                dotR = np.row_stack((dotR, hatR[k]))
                dotY = np.row_stack((dotY, hatY[k]))
            C = np.row_stack((C, k))
        print(dotR, dotY)
        return dotR, dotY

    def gen_delay(self, measure_matrix, nums_prob):
        """
        生成链路时延,生成路径时延，返回测量Y的值
        :param measure_matrix 规约之后的测量矩阵
        :param nums_prob 探测包的数量
        :return:
        """
        assert isinstance(measure_matrix, np.ndarray)
        assert type(nums_prob) == int
        cov = []

        # 每条链路依据其自身的时延方差生成 nums_prob 组时延数据
        delay = []
        for link_var in self.var_vector:
            temp_delay = []

            # 对应于每一条链路生成 nums_prob 组时延数据
            for _ in range(nums_prob):
                temp_delay.append(np.random.normal(0, link_var))

            delay.append(temp_delay)
        delay = np.array(delay)

        # 查看测量矩阵中每一条测量路径包含哪些链路，进而生成该路径的时延数据
        nums_path = len(measure_matrix)
        for row in range(nums_path):
            delay_path = np.zeros(nums_prob)
            for index in np.where(measure_matrix[row] == 1)[0]:
                delay_path = np.add(delay_path, delay[index])
            print("delay path:", delay_path)

            # 计算时延方差
            delay_cov = np.var(delay_path)
            cov.append(delay_cov)

        return np.array(cov)

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


if __name__ == "__main__":
    topo = Topology(None, None, None, [6, 2, 5, 2, 4])
    measure_matrix = [[0, 0, 1, 0, 0],
                     [1, 0, 1, 0, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0]]
    print(topo.gen_delay(np.array(measure_matrix), 10))
