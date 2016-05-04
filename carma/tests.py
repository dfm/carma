# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .conversions import get_roots_from_params, get_alpha_and_beta, sort_params
from ._carma import carma_log_likelihood, carma_psd, carma_covariance

__all__ = [
    "test_carma_psd", "test_carma_covariance", "test_carma_log_likelihood",
]


def python_psd(log_sigma, ar, ma, f):
    alpha, beta = get_alpha_and_beta(
        get_roots_from_params(ar), get_roots_from_params(ma))
    w = 2 * np.pi * 1.0j * f
    return np.exp(2*log_sigma) * (
        np.abs(np.sum(beta[None, :]*w[:, None]**np.arange(len(beta))[None, :],
                      axis=1))**2 /
        np.abs(np.sum(alpha[None, :] *
                      w[:, None]**np.arange(len(alpha))[None, :], axis=1))**2
    )


def python_covariance(log_sigma, ar, ma, tau):
    arroots = get_roots_from_params(ar)
    alpha, beta = get_alpha_and_beta(
        arroots, get_roots_from_params(ma))
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

    K = np.exp(2*log_sigma) * np.sum(terms, axis=-1)
    if not np.all(K.imag < 1e-10):
        raise ValueError("invalid")
    return K.real


def test_carma_psd():
    log_sigma = np.log(0.1)
    arpars = np.log(np.array([0.1, 0.5, 10.0]))
    mapars = np.log(np.array([0.1, 0.5]))

    N = 501
    f = np.linspace(-10, 10, N)
    psd1 = carma_psd(log_sigma, arpars, mapars, f)
    psd2 = python_psd(log_sigma, arpars, mapars, f)

    assert np.allclose(psd1, psd2)


def test_carma_covariance():
    log_sigma = np.log(0.1)
    arpars = np.log(np.array([0.1, 0.5, 10.0]))
    mapars = np.log(np.array([0.1, 0.5]))

    N = 501
    t = np.linspace(-10, 10, N)
    tau = np.abs(t[:, None] - t[None, :])
    c1 = carma_covariance(log_sigma, arpars, mapars, tau)
    c2 = python_covariance(log_sigma, arpars, mapars, tau)

    assert np.allclose(c1, c2)


def test_carma_log_likelihood():
    _carma_log_likelihood(np.log(np.array([0.1, 0.5])))
    _carma_log_likelihood(np.log(np.array([0.1])))
    _carma_log_likelihood(np.array([]))


def _carma_log_likelihood(mapars, seed=1234):
    np.random.seed(seed)

    log_sigma = np.log(0.1)
    arpars = np.log(np.array([0.1, 0.5, 1.0]))

    N = 501
    t = np.linspace(-5, 5, N)
    yerr = 1e-8*np.ones_like(t)
    tau = np.abs(t[:, None] - t[None, :])
    K = carma_covariance(log_sigma, arpars, mapars, tau)
    K[np.diag_indices_from(K)] += yerr**2
    y = np.random.multivariate_normal(np.zeros(len(t)), K)

    ll1 = carma_log_likelihood(log_sigma, arpars, mapars, t, y, yerr)
    ll2 = -0.5 * np.dot(y, np.linalg.solve(K, y))
    ll2 -= 0.5 * np.linalg.slogdet(K)[1]
    ll2 -= 0.5 * N * np.log(2 * np.pi)

    assert np.allclose(ll1, ll2)


def test_sort_params():
    pars = np.log(np.array([0.1, 0.5, 1.0, 0.1, 60.5, 1.23, 0.1]))
    pars2 = sort_params(pars)
    assert pars2[-1] == pars[-1]
    roots = get_roots_from_params(pars2)
    assert np.all(np.diff(roots.imag)[1::2] > 0.0)

    pars = np.log(np.array([0.1, 0.5, 1.0, 0.1, 60.5, 1.23]))
    pars2 = sort_params(pars)
    roots = get_roots_from_params(pars2)
    assert np.all(np.diff(roots.imag)[1::2] > 0.0)

    pars = np.log(np.array([0.1]))
    pars2 = sort_params(pars)
    roots = get_roots_from_params(pars2)
    assert np.all(np.diff(roots.imag)[1::2] > 0.0)

    pars = np.log(np.array([]))
    pars2 = sort_params(pars)
    roots = get_roots_from_params(pars2)
    assert np.all(np.diff(roots.imag)[1::2] > 0.0)
