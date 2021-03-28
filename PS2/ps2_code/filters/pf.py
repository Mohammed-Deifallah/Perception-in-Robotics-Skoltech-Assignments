"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

def low_variance_sampler(M, weights):
    # Follow Pseudocode from ProbRob Page 110
    #Normalize weights: https://robotics.stackexchange.com/a/16786
    new_weights = np.array(weights) / np.sum(weights)
    X = []
    r = uniform(low=0.0, high=1/M)
    c = new_weights[0]
    i = 0
    for m in range(M):
        U = r + m/M
        while U > c:
            i = i + 1
            c = c + new_weights[i]
        X.append(i)
    return X

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)

        # TODO add here specific class variables for the PF
        self.M = num_particles
        self.particles = np.ones(shape=(self.M, 3)) * initial_state.mu[:, 0]
        np.random.seed(42)

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for i in range(self.M):
            self.particles[i] = sample_from_odometry(self.particles[i], u, self._alphas)

        params = get_gaussian_statistics(self.particles)

        self._state_bar.mu = params.mu
        self._state_bar.Sigma = params.Sigma

    def update(self, z):
        # TODO implement correction step
        b, lm_id = z[0], z[1]
        obs_bars = np.array([get_observation(self.particles[i], lm_id)[0] for i in range(self.M)])
        wrp_angles = np.array([wrap_angle(obs_bars[i] - b) for i in range(self.M)])

        weights = gaussian().pdf(wrp_angles / np.sqrt(self._Q))

        self.particles = self.particles[low_variance_sampler(self.M, weights)]
        
        params = get_gaussian_statistics(self.particles)
        self._state.mu = params.mu
        self._state.Sigma = params.Sigma
