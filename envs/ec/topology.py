import numpy as np
from numpy.linalg import matrix_rank as rank, inv as inverse
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
        extend_matrix=np.copy(self.matrix)
        len=self.matrix.shape[0]
        y_array=np.empty(shape=(0,2),dtype=int)#记录y_ij
        for i in range(len):
            y_array=np.append(y_array,np.array([int(i+1),int(i+1)]).reshape(1,2),axis=0)

        for index in range(len):
            for j in range(index+1,len):
                y_array=np.append(y_array,np.array([index+1,j+1]))
                extend_matrix=np.vstack((extend_matrix,(self.matrix[index])&(self.matrix[j])))
        # print(extend_matrix)
        # print(y_array)
        return extend_matrix,y_array


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

    def cal_phi(self,Y_value,reduced_matrix):
        """
        \mathcal{A}_i(\boldsymbol{\theta}) = 2(\sum_{l \in p_i} \theta_l)^2 \sum_{k=1}^{|L|}b_{k,i}^2
        \phi_{i}=\frac{\sqrt{\mathcal{A}_{i}(\boldsymbol{\theta})}}{\sum_{j=1}^{|L|} \sqrt{\mathcal{A}_{j}(\boldsymbol{\theta})}}
        计算A_i,计算\phi_i,
        :return:
        """
        assert isinstance(Y_value,list)
        # assert isinstance(reduced_matrix,np.array)
        #求规约矩阵的逆矩阵
        inverse_matrix = inverse(reduced_matrix)
        A = []
        Phi = []
        for i in range(len(Y_value)):
            #取出逆矩阵的列平方和
            sum_col = 0
            for k in range(len(inverse_matrix[:,i])):
                sum_col += inverse_matrix[:,i][k]**2
            value = 2*(Y_value[i]**2)*sum_col
            A.append(value)
        #计算 A 的平方根和
        sum_A = 0
        for i in range(len(A)):
            sum_A += np.sqrt(A[i])
        for i in range(len(Y_value)):
            Phi.append(np.sqrt(A[i])/sum_A)
        # print("the sum of phi:",sum(Phi))
        return Phi

    def cal_proportions(self,phi_i,dot_y):
        """
        计算H，返回监测节点的比例
        :return:
        """
        #计算pi的权重
        p_i_w=np.empty(shape=(1,0))
        for i in range(self.node_matrix.shape[0]):#根据路径遍历
            y_i_index=set((np.argwhere(dot_y==i+1)[:,0]))#<h_ij或者h_ji>返回dot_y中元素为i+1的行坐标(与顶点i+1有关的路径）
            sum=0
            for j in y_i_index:#对相关路径权重求和
                sum+=phi_i[j]
            sum=sum/2
            p_i_w=np.append(p_i_w,sum)#添加路径权重
        # print(p_i_w)

        #计算节点的权重
        nodes=np.argwhere(self.monitor_vector==1).flatten()+1#获取所有节点编号
        nodes_i_w=np.empty(shape=(1,0))
        for i in list(nodes):#对每一个节点遍历
            sum=0
            i_line=set(np.argwhere(self.node_matrix[:,i-1]==1).flatten()+1)#找出和顶点i有关的路径
            for j in i_line:
                sum+=p_i_w[j-1]
            nodes_i_w=np.append(nodes_i_w,sum/2)
        # print(nodes_i_w)
        return nodes_i_w

    def get_proportion(self):
        pass

if __name__ == "__main__":
    R = np.array([[0., 0., 1. ,0., 0.],
 [0., 0. ,1., 1. ,0.],
 [0. ,1., 1., 0. ,0.],
 [0. ,0. ,1. ,0. ,1.],
 [1. ,0. ,1., 0. ,0.]])
    topo = Topology(0,0,0)
    topo.cal_phi([0.2,0.2,0.2,0.2,0.2],R)