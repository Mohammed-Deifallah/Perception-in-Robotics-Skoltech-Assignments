"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

def G_t(state, u):
    theta = state[2]
    drot1, dtrans = u[0], u[1]
    return np.array([[1, 0, -dtrans * np.sin(theta + drot1)],
                    [0, 1, dtrans * np.cos(theta + drot1)],
                    [0, 0, 1]])

def V_t(state, u):
    theta = state[2]
    drot1, dtrans = u[0], u[1]
    return np.array([[-dtrans * np.sin(theta + drot1), np.cos(theta + drot1), 0],
                    [dtrans * np.cos(theta + drot1), np.sin(theta + drot1), 0],
                    [1, 0, 1]])

def H_t(state, lm_id, field_map):
    x, y = state[0], state[1]
    lm_id = int(lm_id)
    mu_x, mu_y = field_map.landmarks_poses_x[lm_id], field_map.landmarks_poses_y[lm_id]
    dx, dy = mu_x - x, mu_y - y
    q = np.power(dx, 2) + np.power(dy, 2)
    return np.array([[dy/q, -dx/q, -1]])
    
class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self.mu[2] = wrap_angle(self.mu[2]) #auxiliary step
        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T
        M = get_motion_noise_covariance(u, self._alphas)
        G = G_t(self.mu, u)
        V = V_t(self.mu, u)

        self._state_bar.Sigma = np.dot(G, np.dot(self.Sigma, G.T)) + np.dot(V, np.dot(M, V.T))

    def update(self, z):
        # TODO implement correction step
        self.mu_bar[2] = wrap_angle(self.mu_bar[2]) #auxiliary step
        #Following pseudo-code L06 page 4, ProbRob 204
        b = z[0]
        lm_id = z[1]
        b_bar = get_expected_observation(self.mu_bar, lm_id)[0]
        H = H_t(self.mu_bar, lm_id, self._field_map)
        S = np.dot(H, np.dot(self.Sigma_bar, H.T)) + self._Q
        S_inv = np.linalg.inv(S)
        K = np.dot(self.Sigma_bar, np.dot(H.T, S_inv))
        self._state_bar.mu += K * (b - b_bar)
        I = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        self._state_bar.Sigma = np.dot(I - np.dot(K, H), self._state_bar.Sigma)

        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
