# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np
import matplotlib.pyplot as pl
from numpy.polynomial.polynomial import polyfromroots
from ._carma import carma_log_likelihood, carma_psd, carma_covariance

__all__ = []


def _get_alpha_and_beta(a, b):
    alpha = polyfromroots(a)
    beta = polyfromroots(b)
    return alpha, beta / beta[0]


def python_psd(sigma, arroots, maroots, f):
    alpha, beta = _get_alpha_and_beta(arroots, maroots)
    w = 2 * np.pi * 1.0j * f
    return sigma**2 * (
        np.abs(np.sum(beta[None, :]*w[:, None]**np.arange(len(beta))[None, :],
                      axis=1))**2 /
        np.abs(np.sum(alpha[None, :] *
                      w[:, None]**np.arange(len(alpha))[None, :], axis=1))**2
    )


def python_covariance(sigma, arroots, maroots, tau):
    alpha, beta = _get_alpha_and_beta(arroots, maroots)
    p = len(alpha) - 1
    q = len(beta) - 1

    norm = np.sum(beta[None, :]*arroots[:, None]**np.arange(q+1)[None, :],
                  axis=1)
    norm *= np.sum(beta[None, :]*(-arroots[:, None])**np.arange(q+1)[None, :],
                   axis=1)
    norm /= -2.0 * arroots.real

    for k in range(p):
        l = np.arange(p) != k
        norm[k] /= np.prod((arroots[l] - arroots[k]) *
                           (np.conjugate(arroots[l]) + arroots[k]))

    terms = norm[None, None, :]*np.exp(arroots[None, None, :]*tau[:, :, None])

    K = sigma**2 * np.sum(terms, axis=-1)
    if not np.all(K.imag < 1e-10):
        raise ValueError("invalid")
    return K.real


def test_carma_psd():
    sigma = 0.1
    a = np.array([-0.5 + 10j, -0.5 - 10j, -1.0 + 0.1j, -1.0 - 0.1j])
    b = np.array([-0.5 + 0.1j, -0.5 - 0.1j])

    N = 501
    f = np.linspace(-10, 10, N)
    psd1 = carma_psd(sigma, a, b, f)
    psd2 = python_psd(sigma, a, b, f)

    assert np.allclose(psd1, psd2)


def test_carma_covariance():
    sigma = 0.1
    a = np.array([-1.0, -0.5 + 10j, -0.5 - 10j, -1.0 + 0.1j, -1.0 - 0.1j])
    b = np.array([-0.5 + 0.1j, -0.5 - 0.1j, -1 + 0.1j, -1 - 0.1j])

    N = 501
    t = np.linspace(-10, 10, N)
    tau = np.abs(t[:, None] - t[None, :])
    c1 = carma_covariance(sigma, a, b, tau)
    c2 = python_covariance(sigma, a, b, tau)

    assert np.allclose(c1, c2)


def test_carma_log_likelihood(seed=1234):
    np.random.seed(seed)

    sigma = 0.1
    a = np.array([-1.0, -0.1+2.0j, -0.1-2.0j], dtype=np.complex128)
    b = np.array([-0.05+0.01j, -0.05-0.01j], dtype=np.complex128)

    N = 501
    t = np.linspace(-5, 5, N)
    yerr = 1e-8*np.ones_like(t)
    tau = np.abs(t[:, None] - t[None, :])
    K = carma_covariance(sigma, a, b, tau)
    K[np.diag_indices_from(K)] += yerr**2
    y = np.random.multivariate_normal(np.zeros(len(t)), K)

    ll1 = carma_log_likelihood(sigma, a, b, t, y, yerr)
    ll2 = -0.5 * np.dot(y, np.linalg.solve(K, y))
    ll2 -= 0.5 * np.linalg.slogdet(K)[1]
    ll2 -= 0.5 * N * np.log(2 * np.pi)

    assert np.allclose(ll1, ll2)
