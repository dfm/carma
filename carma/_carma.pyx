# distutils: language = c++
from __future__ import division

cimport cython
from libcpp.complex cimport complex

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
CDTYPE = np.complex128
ctypedef np.complex128_t CDTYPE_t

cdef extern from "carma.h" namespace "carma":
    cdef double log_likelihood(
        double sigma, unsigned p, complex[double]* ar, unsigned q, complex[double]* ma,
        unsigned n, double* t, double* y, double* yerr
    )

    cdef double psd(
        double sigma, unsigned p, complex[double]* ar, unsigned q, complex[double]* ma,
        unsigned n, double* f, double* out
    )

    cdef double covariance(
        double sigma, unsigned p, complex[double]* ar, unsigned q, complex[double]* ma,
        unsigned n, double* tau, double* out
    )


def check_roots(
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] arroots,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] maroots,
):
    if len(maroots) >= len(arroots):
        raise ValueError("q must be less than p")

    if not np.all(arroots.real < 0.0):
        raise ValueError("arroots must have negative real parts")
    if not len(set(arroots) & set(np.conjugate(arroots))) == len(arroots):
        raise ValueError("arroots must have all conjugate pairs")

    if not np.all(maroots.real < 0.0):
        raise ValueError("maroots must have negative real parts")
    if not len(set(maroots) & set(np.conjugate(maroots))) == len(maroots):
        raise ValueError("maroots must have all conjugate pairs")


def carma_log_likelihood(
    double sigma,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] arroots,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] maroots,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] yerr
):
    check_roots(arroots, maroots)

    if len(x) != len(y) or len(y) != len(yerr):
        raise ValueError("dimension mismatch")

    return log_likelihood(
        sigma, len(arroots), <complex[double]*>arroots.data,
        len(maroots), <complex[double]*>maroots.data,
        len(x), <double*>x.data, <double*>y.data, <double*>yerr.data
    )


def carma_psd(
    double sigma,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] arroots,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] maroots,
    f
):
    check_roots(arroots, maroots)
    cdef np.ndarray[DTYPE_t] f_array = np.atleast_1d(f).ravel()
    cdef np.ndarray[DTYPE_t] values = np.empty_like(f_array)
    psd(sigma, len(arroots), <complex[double]*>arroots.data,
        len(maroots), <complex[double]*>maroots.data,
        f_array.size, <double*>f_array.data, <double*>values.data)
    return values.reshape(f.shape)


def carma_covariance(
    double sigma,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] arroots,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] maroots,
    tau
):
    check_roots(arroots, maroots)

    cdef np.ndarray[DTYPE_t] tau_array = np.atleast_1d(tau).ravel()
    cdef np.ndarray[DTYPE_t] values = np.empty_like(tau_array)
    covariance(sigma, len(arroots), <complex[double]*>arroots.data,
               len(maroots), <complex[double]*>maroots.data,
               tau_array.size, <double*>tau_array.data, <double*>values.data)
    return values.reshape(tau.shape)
