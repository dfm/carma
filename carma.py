# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from numpy.polynomial.polynomial import polyfromroots

__all__ = ["CARMA"]


class CARMA(object):

    def __init__(self, sigma, arroots, maroots):
        self.arroots = np.atleast_1d(arroots).astype(np.complex128)
        self.maroots = np.atleast_1d(maroots).astype(np.complex128)

        self.p = len(arroots)
        self.q = len(maroots)

        if self.q >= self.p:
            raise ValueError("q must be less than p")
        if not np.all(self.arroots.real < 0.0):
            raise ValueError("arroots must have a negative real part")
        if not np.all(self.maroots.real <= 0.0):
            raise ValueError("maroots must have a negative real part")

        self.alpha = polyfromroots(self.arroots)
        self.alpha /= self.alpha[-1]
        self.beta = polyfromroots(self.maroots)
        self.beta /= self.beta[0]
        self.b = np.concatenate((self.beta, np.zeros(self.p - self.q - 1)))

        U = np.atleast_1d(arroots)[None, :] ** np.arange(self.p)[:, None]
        self.b = np.dot(self.b, U)

        e = np.zeros(self.p, dtype=np.complex128)
        e[-1] = sigma
        J = np.linalg.solve(U, e)

        self.V = -J[:, None] * np.conjugate(J)[None, :]
        self.V /= (self.arroots[:, None] + np.conjugate(self.arroots)[None, :])

        self.reset()

    def reset(self):
        # Step 2
        self.x = np.zeros(self.p, dtype=np.complex128)
        self.P = np.array(self.V)

    def update(self, y):
        # Steps 10-12
        K = np.dot(self.P, np.conjugate(self.b)) / self.Vary
        self.x += (y - self.Ey) * K
        self.P -= self.Vary * K[:, None] * np.conjugate(K)[None, :]

    def advance(self, dt):
        # Steps 7-8
        lam = np.exp(self.arroots * dt)
        self.x *= lam
        self.P = lam[:, None] * (self.P - self.V) * np.conjugate(lam)[None, :]
        self.P += self.V

    def predict(self, yerr):
        # Step 3 and step 9
        self.Ey = np.dot(self.b, self.x)
        self.Ey = self.Ey.real
        self.Vary = yerr**2
        self.Vary += np.dot(self.b, np.dot(self.P, np.conjugate(self.b))).real
        return self.Ey, self.Vary

    def log_likelihood(self, t, y, yerr):
        n = len(t)
        Ey = np.empty(n)
        Vary = np.empty(n)

        self.reset()
        for i in range(n):
            Ey[i], Vary[i] = self.predict(yerr[i])
            if Vary[i] < 0.0:
                print("unstable")
                return -np.inf
            self.update(y[i])
            if i < n-1:
                self.advance(t[i+1] - t[i])

        resid2 = (y - Ey)**2
        ll = n * np.log(2 * np.pi) + np.sum(np.log(Vary) + resid2 / Vary)
        return -0.5 * ll


def get_alpha_and_beta(arroots, maroots):
    alpha = polyfromroots(arroots)
    beta = polyfromroots(maroots)
    beta /= beta[0]
    return alpha, beta


def carma_psd(sigma, arroots, maroots, f):
    w = 2j * np.pi * f
    alpha, beta = get_alpha_and_beta(arroots, maroots)

    p = len(alpha)
    q = len(beta)

    num = np.sum(beta[None, :]*w[:, None] ** np.arange(q)[None, :], axis=1)
    denom = np.sum(alpha[None, :]*w[:, None] ** np.arange(p)[None, :], axis=1)

    return (sigma * np.abs(num) / np.abs(denom)) ** 2


def carma_covar(sigma, arroots, maroots, tau):
    alpha, beta = get_alpha_and_beta(arroots, maroots)
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


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as pl
    np.random.seed(1234)

    sigma = 0.1

    # freqs = [1.0, 2.0, 0.01]
    # a = get_arroots(freqs, [1.0, 2.0, 0.1])

    # b = -np.random.rand(1) - 1.j * np.random.rand(1)
    # b = np.array(list(set(list(np.append(b, np.conjugate(b))))))

    a = np.array([-0.5 + 0.01j, -0.5 - 0.01j, -1.0 + 0.1j, -1.0 - 0.1j])
    b = np.array([-0.5 + 0.1j, -0.5 - 0.1j])
    # b = np.array([1.0, 0.5])

    # print(a)
    # print(b)

    # x = np.sort(np.random.rand(1000))
    # x = np.linspace(0, 1, 50)
    x = np.array([0.5, 0.6, 10.0])
    yerr = 0.1 * np.ones_like(x)
    y = 0.1 * x

    strt = time.time()
    model = CARMA(sigma, a, b)
    ll1 = model.log_likelihood(x, y, yerr)
    print(ll1)
    assert 0
    print(time.time() - strt)

    strt = time.time()
    K = carma_covar(sigma, a, b, np.abs(x[:, None] - x[None, :]))
    K[np.diag_indices_from(K)] += yerr**2
    ll2 = -0.5 * np.dot(y, np.linalg.solve(K, y))
    ll2 -= 0.5 * np.linalg.slogdet(K)[1]
    ll2 -= 0.5 * len(x) * np.log(2*np.pi)
    print(time.time() - strt)

    print(ll1 - ll2)

    pl.clf()
    tau = np.atleast_2d(np.linspace(0, 10, 10000))
    p = carma_covar(sigma, a, b, tau)
    pl.plot(tau[0], p[0])
    pl.savefig("plot2.png")

    f = np.linspace(-10, 10, 10000)
    p = carma_psd(sigma, a, b, f)
    pl.plot(f, p)
    # pl.semilogy(f, p)
    # for _ in freqs:
    #     pl.gca().axvline(_)
    pl.savefig("plot1.png")
