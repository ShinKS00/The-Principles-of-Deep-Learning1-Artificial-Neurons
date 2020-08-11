# import nodes
# import numpy as np
# class Affine_Function:
#     def __init__(self):
#         self._feature_dim = feature_dim
#         self._Th = Th
#         self._Z1_list = [None]*(self._feature_dim + 1)
#         self._Z2_list = self._Z1_list.copy()
#         self._dZ1_list, self._dZ2_list = self._Z1_list.copy(), self._Z1_list.copy()
#         self._dTh_list = self._Z1_list.copy()
        
#         self.affine_imp()
#     def affine_imp(self):
#         self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
#         self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]
    
#     def forward(self, X):
#         for node_idx in range(1, self._feature_dim + 1):
#             self._Z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:node_idx])
        
#         self._Z2_list[1] = self._node2[1].forward(self._Th[0], self._Z1_list[1])
#         for node_idx in range(2, feature_dim + 1):
#             self._Z2_list[node_idx] = self._node2[node_idx].forward(self._Z2_list[node_idx-1], self._Z1_list[node_idx])
#         return self._Z2_list[-1]
    
#     def backward(self, dZ2_last, lr):
#         self._dZ2_ist[-1] = dZ2_last
        
#         for node_idx in reversed(range(1, self._feature_dim + 1)):
#             dZ2, dZ1 = self._node2[node_idx].backward(self._dZ1_list[node_idx])
#             self._dZ2_list[node_idx-1] = dZ2
#             self._dZ1_list[node_idx] = dZ1
        
#         self._dTh_list[0] = self._dZ2_list[0]
#         for node_idx in reversed(range(1, self._feature_dim + 1)):
#             dTh, _ = self._node1[node_idx].backward(self._dZ1_list[node_idx])
#             self._dTh_list[node_idx] = dTh
        
#         for th_idx in range(self._Th.shape[0]):
#             self._Th[th_idx] = self._Th[th_idx] - lr*np.sum(self._dTh_list[th_idx])
        
#         return self._Th
        
        
        
# class MSE_Cost:
#     def __init__(self):
#         self.cost_imp()
        
#     def cost_imp(self):
#         self._node3 = nodes.minus_node()
#         self._node4 = nodes.square_node()
#         self._node5 = nodes.mean_node()
    
#     def forward(self, Y, Pred):
#         Z3 = self._node3.forward(Y, Pred)
#         Z4 = self._node4.forward(Z3)
#         J = self._node5.forward(Z4)
#         return J
#     def backward(self):
#         dZ4 = self._node5.backward(1)
#         dZ3 = self._node4.backward(dZ4)
#         _, dZ2_last = self._node3.backward(dZ3)
#         return dZ2_last
    
#%%    
    
import basic_nodes as nodes
import numpy as np
class Affine_Function:
    def __init__(self,feature_dim, Th):
        self._feature_dim = feature_dim
        self._Th = Th
        self._Z1_list = [None]*(self._feature_dim + 1)
        self._Z2_list = self._Z1_list.copy()
        self._dZ1_list, self._dZ2_list = self._Z1_list.copy(), self._Z1_list.copy()
        self._dTh_list = self._Z1_list.copy()
        
        self.affine_imp()
    def affine_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]
    
    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._Z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:,node_idx])
        
        self._Z2_list[1] = self._node2[1].forward(self._Th[0], self._Z1_list[1])
        for node_idx in range(2, self._feature_dim + 1):
            self._Z2_list[node_idx] = self._node2[node_idx].forward(self._Z2_list[node_idx-1], self._Z1_list[node_idx])
        return self._Z2_list[-1]
    
    def backward(self, dZ2_last, lr):
        self._dZ2_list[-1] = dZ2_last
        
        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dZ2, dZ1 = self._node2[node_idx].backward(self._dZ2_list[node_idx])
            self._dZ2_list[node_idx-1] = dZ2
            self._dZ1_list[node_idx] = dZ1
        
        self._dTh_list[0] = self._dZ2_list[0]
        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dTh, _ = self._node1[node_idx].backward(self._dZ1_list[node_idx])
            self._dTh_list[node_idx] = dTh
        
        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] = self._Th[th_idx] - lr*np.sum(self._dTh_list[th_idx])
        
        return self._Th
        
        
        
class MSE_Cost:
    def __init__(self):
        self.cost_imp()
        
    def cost_imp(self):
        self._node3 = nodes.minus_node()
        self._node4 = nodes.square_node()
        self._node5 = nodes.mean_node()
    
    def forward(self, Y, Pred):
        Z3 = self._node3.forward(Y, Pred)
        Z4 = self._node4.forward(Z3)
        J = self._node5.forward(Z4)
        return J
    def backward(self):
        dZ4 = self._node5.backward(1)
        dZ3 = self._node4.backward(dZ4)
        _, dZ2_last = self._node3.backward(dZ3)
        return dZ2_last
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        