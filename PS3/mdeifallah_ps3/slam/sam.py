"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from scipy.sparse.linalg import inv as sp_inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian, observation_jacobian

class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        
        
        self.graph = mrob.FGraph()
        self.xn = np.squeeze(initial_state.mu)
        self.lm_ids = {}
        self.xn_id = self.graph.add_node_pose_2d(self.xn)
        self.graph.add_factor_1pose_2d(self.xn, self.xn_id, inv(initial_state.Sigma))
        #self.graph.print(True) # --> Task 1.A
        

    def predict(self, u):
        prev_list = self.graph.get_estimated_state()
        #print('State before: ', prev_list) # --> Task 1.B
        x = np.zeros(3)
        x_id = self.graph.add_node_pose_2d(x)
        M = get_motion_noise_covariance(u, self.alphas)
        _, V = state_jacobian(self.xn, u)
        W_u = inv(np.dot(V, np.dot(M, V.T)))
        self.graph.add_factor_2poses_2d_odom(u, self.xn_id, x_id, W_u)
        cur_list = self.graph.get_estimated_state()
        #print('State after: ', cur_list) # --> Task 1.B
        self.xn_id = x_id
        self.xn = np.squeeze(cur_list[-1])

    def update(self, z):
        W_z = inv(self.Q)
        nodes = self.graph.get_estimated_state()
        for z_i in z:
            if z_i[2] in self.lm_ids:
                lm_id = self.lm_ids[z_i[2]]
                new = False
            else:
                lm_id = self.graph.add_node_landmark_2d(np.zeros(2))
                new = True
            lm_id = int(lm_id)
            self.lm_ids[z_i[2]] = lm_id
            self.graph.add_factor_1pose_1landmark_2d(z_i[:2], self.xn_id, lm_id, W_z, initializeLandmark=new)
        
        #print(self.graph.get_estimated_state()) # --> Task 1.C
             
    def solve(self, gn=True):
        if gn:
            self.graph.solve()
        else:
            self.graph.solve(method=mrob.LM)
        #self.graph.print(True) # --> Task 1.D
        
    def chi2(self):
        return self.graph.chi2()
    
    def get_adj_matrix(self):
        return self.graph.get_adjacency_matrix()
    
    def get_info_matrix(self):
        return self.graph.get_information_matrix()
    
    def get_cov_matrix(self):
        return sp_inv(self.get_info_matrix())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
