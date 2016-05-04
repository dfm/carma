# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .conversions import sort_params, get_roots_from_params
from ._carma import carma_log_likelihood, carma_covariance, carma_psd

__all__ = ["CARMAModel"]


class CARMAModel(object):

    def __init__(self, log_sigma, arpars, mapars, log_jitter=-10,
                 max_R0=np.inf):
        self.log_jitter = log_jitter
        self.log_sigma = log_sigma
        self.p = len(arpars)
        self.arpars = sort_params(arpars)
        self.q = len(mapars)
        self.mapars = sort_params(mapars)
        self.max_R0 = max_R0

    def log_prior(self):
        if not (-12 < self.log_jitter < 12 and -12 < self.log_sigma < 12):
            return -np.inf
        if np.any(self.arpars < -12) or np.any(self.arpars > 12):
            return -np.inf
        if np.any(self.mapars < -12) or np.any(self.mapars > 12):
            return -np.inf
        roots = get_roots_from_params(self.arpars)
        if not np.all(np.diff(roots.imag)[1::2] > 0.0):
            return -np.inf
        roots = get_roots_from_params(self.mapars)
        if not np.all(np.diff(roots.imag)[1::2] > 0.0):
            return -np.inf
        R0 = carma_covariance(self.log_sigma, self.arpars, self.mapars, 0.0)
        if R0 > self.max_R0:
            return -np.inf
        return 0.0

    def log_likelihood(self, t, y, yerr=None):
        if yerr is None:
            yerr = np.exp(self.log_jitter) + np.zeros_like(t)
        else:
            yerr = np.sqrt(np.exp(2*self.log_jitter) + yerr**2)

        try:
            ll = carma_log_likelihood(self.log_sigma, self.arpars, self.mapars,
                                      t, y, yerr)
        except (RuntimeError, ValueError):
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return ll

    def log_probability(self, t, y, yerr=None):
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return self.log_likelihood(t, y, yerr=yerr) + lp

    def __call__(self, theta, t, y, yerr=None):
        self.set_vector(theta)
        return self.log_probability(t, y, yerr)

    def cost(self, theta, t, y, yerr=None):
        self.set_vector(theta)
        return -self.log_probability(t, y, yerr)

    def get_vector(self):
        return np.concatenate(([self.log_jitter],
                               self.arpars, self.mapars))

    def set_vector(self, vector):
        self.log_jitter = vector[0]
        # self.log_sigma = vector[1]
        self.arpars = vector[1:1+self.p]
        self.mapars = vector[1+self.p:1+self.p+self.q]

    def sample(self, t):
        tau = t[:, None] - t[None, :]
        K = carma_covariance(self.log_sigma, self.arpars, self.mapars, tau)
        K[np.diag_indices_from(K)] += np.exp(2*self.log_jitter)
        return np.random.multivariate_normal(np.zeros_like(t), K)

    def psd(self, f):
        return carma_psd(self.log_sigma, self.arpars, self.mapars, f)
