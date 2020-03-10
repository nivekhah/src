import numpy as np
from numpy.linalg import matrix_rank as rank, inv as inverse
from numpy.linalg import solve


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
        self.proportion = self.get_proportion()

    def get_extend_matrix(self):
        """
        获取扩展测量矩阵,返回扩展的测量矩阵
        :return:
        """
        extend_matrix = np.copy(self.matrix)
        len = self.matrix.shape[0]
        y_array = np.empty(shape=(0, 2), dtype=int)  # 记录y_ij
        for i in range(len):
            y_array = np.append(y_array, np.array([int(i + 1), int(i + 1)]).reshape(1, 2), axis=0)

        for index in range(len):
            for j in range(index + 1, len):
                y_array = np.append(y_array, np.array([index + 1, j + 1]).reshape(1, 2), axis=0)
                extend_matrix = np.vstack((extend_matrix, (self.matrix[index]) & (self.matrix[j])))
        # print(extend_matrix)
        # print(y_array)
        return extend_matrix, y_array

    def matrix_reduction(self, hatR, hatY):
        """
        对测量矩阵规约，返回规约的矩阵和对应Y
        :return:
        """
        dotR = []
        dotY = np.empty(shape=[0, 2])
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

    def gen_delay(self, measure_matrix, path_scale, total_nums):
        """
        生成链路时延,生成路径时延，返回测量Y的值
        :param measure_matrix 规约之后的测量矩阵
        :param path_scale 探测包的数量. [0.2, 0.2, 0.2, 0.2, 0.2]
        :param total_nums
        :return:
        """
        assert isinstance(measure_matrix, np.ndarray)
        assert type(total_nums) == int

        cov = []

        for path_index, scale in enumerate(path_scale):
            delay = []
            path_prob_nums = int(total_nums * scale)
            # print("path ", (path_index + 1), " numbers of prob: ", path_prob_nums)
            for link_index in np.where(measure_matrix[path_index] == 1)[0]:
                link_delay = []
                for _ in range(path_prob_nums):
                    link_delay.append(np.random.normal(0, np.sqrt(self.var_vector[link_index])))
                # print("link delay of link ", (link_index + 1), ": ", link_delay)
                delay.append(link_delay)
            # print("all link delay: ", delay)

            path_delay = np.zeros(path_prob_nums)
            for link_delay in delay:
                path_delay = np.add(path_delay, link_delay)
            # print("path delay: ", path_delay)

            delay_cov = np.var(path_delay)
            # print("path delay covariance: ", delay_cov)
            cov.append(delay_cov)

        # print("all path delay covariance", cov)
        return cov

    def cal_phi(self, Y_value, reduced_matrix):
        """
        \mathcal{A}_i(\boldsymbol{\theta}) = 2(\sum_{l \in p_i} \theta_l)^2 \sum_{k=1}^{|L|}b_{k,i}^2
        \phi_{i}=\frac{\sqrt{\mathcal{A}_{i}(\boldsymbol{\theta})}}{\sum_{j=1}^{|L|} \sqrt{\mathcal{A}_{j}(\boldsymbol{\theta})}}
        计算A_i,计算\phi_i,
        :return:
        """
        assert isinstance(Y_value, list)
        # assert isinstance(reduced_matrix,np.array)
        # 求规约矩阵的逆矩阵
        inverse_matrix = inverse(reduced_matrix)
        A = []
        Phi = []
        for i in range(len(Y_value)):
            # 取出逆矩阵的列平方和
            sum_col = 0
            for k in range(len(inverse_matrix[:, i])):
                sum_col += inverse_matrix[:, i][k] ** 2
            value = 2 * (Y_value[i] ** 2) * sum_col
            A.append(value)
        # 计算 A 的平方根和
        sum_A = 0
        for i in range(len(A)):
            sum_A += np.sqrt(A[i])
        for i in range(len(Y_value)):
            Phi.append(np.sqrt(A[i]) / sum_A)
        # print("the sum of phi:",sum(Phi))
        return Phi

    def cal_proportions(self, phi_i, dot_y):
        """
        计算H，返回监测节点的比例
        :return:
        """
        # 计算pi的权重
        p_i_w = np.empty(shape=(1, 0))
        for i in range(self.node_matrix.shape[0]):  # 根据路径遍历
            y_i_index = set((np.argwhere(dot_y == i + 1)[:, 0]))  # <h_ij或者h_ji>返回dot_y中元素为i+1的行坐标(与顶点i+1有关的路径）
            sum = 0
            for j in y_i_index:  # 对相关路径权重求和
                sum += phi_i[j]
            sum = sum / 2
            p_i_w = np.append(p_i_w, sum)  # 添加路径权重
        # print(p_i_w)

        # 计算节点的权重
        nodes = np.argwhere(self.monitor_vector == 1).flatten() + 1  # 获取所有节点编号
        nodes_i_w = np.empty(shape=(1, 0))
        for i in list(nodes):  # 对每一个节点遍历
            sum = 0
            i_line = set(np.argwhere(self.node_matrix[:, i - 1] == 1).flatten() + 1)  # 找出和顶点i有关的路径
            for j in i_line:
                sum += p_i_w[j - 1]
            nodes_i_w = np.append(nodes_i_w, sum / 2)
        # print(nodes_i_w)
        return nodes_i_w

    def get_proportion(self):
        """
        调用函数取得比列
        :return:
        """
        hatR, hatY = self.get_extend_matrix()
        dotR, dotY = self.matrix_reduction(hatR, hatY)
        self.reduced_matrix = dotR
        Y_value = self.gen_delay(dotR, [0.2, 0.2, 0.2, 0.2, 0.2], 10000)
        Phi = self.cal_phi(Y_value, dotR)
        self.Phi = Phi
        proportion = self.cal_proportions(Phi, dotY)
        return proportion

    def cal_measured_link_parameter(self,path_cov):
        """
        计算测量的路径方差
        :param path_cov:
        :return:
        """
        assert isinstance(path_cov,list)
        hatR, hatY = self.get_extend_matrix()
        dotR, _ = self.matrix_reduction(hatR, hatY)
        measured_X = solve(dotR,path_cov)
        return measured_X

def region_error():
     """
    画出均匀分布区间和估计误差
     :return:
     """
     regions = [0,2,4,6,8,10] #设定均匀分布的区间长度值
     center_point = 5 #设置均匀分布的中间值
     k = 100
     fim_errors = []
     average_errors = []
     for region in regions:
         fim_error = np.array([0,0,0,0,0],dtype=float)
         average_error = np.array([0,0,0,0,0],dtype=float)
         for _ in range(k):
             # 1.均匀分布生成方差
             var_vector = np.random.uniform(center_point-region/2,center_point+region/2,5)
             # 2.构建topo
             measure_matrix = np.array([[1, 0, 1, 1, 0],
                                        [1, 0, 1, 0, 1],
                                        [0, 1, 1, 1, 0],
                                        [0, 1, 1, 0, 1]]) #测量矩阵
             node_matrix = np.array([[1, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 0, 1]])  #节点和路径的关系矩阵
             monitor_vector = np.array([1, 1, 0, 0, 1, 1]) #监控节点向量
             topo = Topology(measure_matrix, node_matrix, monitor_vector, var_vector)
             # 3.生成数据，并且推断
             #FIM值
             fim_cov = topo.gen_delay(topo.reduced_matrix,topo.Phi,10000)
             fim_measured_X = topo.cal_measured_link_parameter(fim_cov)
             average_cov = topo.gen_delay(topo.reduced_matrix,[0.2,0.2,0.2,0.2,0.2],10000)
             average_measured_X = topo.cal_measured_link_parameter(average_cov)
             fim_error += np.array(fim_measured_X)-var_vector
             average_error += np.array(average_measured_X)-var_vector
         fim_errors.append(fim_error.sum()/k)
         average_errors.append(average_error.sum()/k)

if __name__ == "__main__":
    # measure_matrix = np.array([[1,0,1,1,0],
    #     [1,0,1,0,1],
    #     [0,1,1,1,0],
    #     [0,1,1,0,1]])
    # node_matrix = np.array([[1,0,0,0,1,0],
    #     [1,0,0,0,0,1],
    #     [0,1,0,0,1,0],
    #     [0,1,0,0,0,1]])
    # monitor_vector = np.array([1,1,0,0,1,1])
    #
    # var_vector = [6, 2, 5, 2, 4]
    # topo = Topology(measure_matrix, node_matrix, monitor_vector, var_vector)
    region_error()
