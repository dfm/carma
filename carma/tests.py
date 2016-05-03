# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from numpy.polynomial.polynomial import polyfromroots
from ._carma import carma_log_likelihood, carma_psd, carma_covariance

__all__ = []


def _get_roots(a):
    a = np.exp(a)
    if len(a) % 2 == 1:
        last_root = [-a[-1]]
        a = a[:-1]
    else:
        last_root = []

    c = a[::2]
    b = a[1::2]
    arg = b*b - 4*c
    arg = np.sqrt(np.abs(arg)) * (1.0*(arg >= 0.0) + 1.0j*(arg < 0.0))
    roots = [-0.5*b[i] + (j-0.5) * arg[i] for i in range(len(b))
             for j in range(2)] + last_root
    return np.array(roots)


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
    arpars = np.log(np.array([0.1, 0.5, 10.0]))
    mapars = np.log(np.array([0.1, 0.5]))
    a = _get_roots(arpars)
    b = _get_roots(mapars)

    N = 501
    f = np.linspace(-10, 10, N)
    psd1 = carma_psd(sigma, arpars, mapars, f)
    psd2 = python_psd(sigma, a, b, f)

    assert np.allclose(psd1, psd2)


def test_carma_covariance():
    sigma = 0.1
    arpars = np.log(np.array([0.1, 0.5, 10.0]))
    mapars = np.log(np.array([0.1, 0.5]))
    a = _get_roots(arpars)
    b = _get_roots(mapars)
    # a = np.array([-1.0, -0.5 + 10j, -0.5 - 10j, -1.0 + 0.1j, -1.0 - 0.1j])
    # b = np.array([-0.5 + 0.1j, -0.5 - 0.1j, -1 + 0.1j, -1 - 0.1j])

    N = 501
    t = np.linspace(-10, 10, N)
    tau = np.abs(t[:, None] - t[None, :])
    c1 = carma_covariance(sigma, arpars, mapars, tau)
    c2 = python_covariance(sigma, a, b, tau)

    assert np.allclose(c1, c2)


def test_carma_log_likelihood(seed=1234):
    np.random.seed(seed)

    sigma = 0.1
    arpars = np.log(np.array([0.1, 0.5, 1.0]))
    mapars = np.log(np.array([0.1, 0.5]))

    N = 501
    t = np.linspace(-5, 5, N)
    yerr = 1e-8*np.ones_like(t)
    tau = np.abs(t[:, None] - t[None, :])
    K = carma_covariance(sigma, arpars, mapars, tau)
    K[np.diag_indices_from(K)] += yerr**2
    y = np.random.multivariate_normal(np.zeros(len(t)), K)

    ll1 = carma_log_likelihood(sigma, arpars, mapars, t, y, yerr)
    ll2 = -0.5 * np.dot(y, np.linalg.solve(K, y))
    ll2 -= 0.5 * np.linalg.slogdet(K)[1]
    ll2 -= 0.5 * N * np.log(2 * np.pi)

    assert np.allclose(ll1, ll2)
