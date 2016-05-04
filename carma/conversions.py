# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from numpy.polynomial.polynomial import polyfromroots

__all__ = ["get_roots_from_params", "get_alpha_and_beta"]


def get_roots_from_params(a):
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
    roots = [-0.5*b[i] + (0.5-j) * arg[i] for i in range(len(b))
             for j in range(2)] + last_root
    return np.array(roots, dtype=np.complex128)


def sort_params(params):
    if len(params) <= 2:
        return params
    roots = get_roots_from_params(params)
    inds = 2*np.argsort(-np.abs(roots.imag[::2]))
    inds = np.array([i+j for i in inds for j in range(2)])[:len(roots)]
    return params[inds]


def get_alpha_and_beta(a, b):
    alpha = polyfromroots(a)
    beta = polyfromroots(b)
    return alpha, beta / beta[0]
