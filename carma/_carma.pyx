# distutils: language = c++
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "carma.h" namespace "carma":
    cdef double log_likelihood(
        double log_sigma, unsigned p, double* ar, unsigned q, double* ma,
        unsigned n, double* t, double* y, double* yerr
    ) except +

    cdef double psd(
        double log_sigma, unsigned p, double* ar, unsigned q, double* ma,
        unsigned n, double* f, double* out
    ) except +

    cdef double covariance(
        double log_sigma, unsigned p, double* ar, unsigned q, double* ma,
        unsigned n, double* tau, double* out
    ) except +


def check_pars(
    np.ndarray[DTYPE_t, ndim=1, mode='c'] arpars,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] mapars,
):
    if len(mapars) >= len(arpars):
        raise ValueError("q must be less than p")


def carma_log_likelihood(
    double log_sigma,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] arpars,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] mapars,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] yerr
):
    check_pars(arpars, mapars)

    if len(x) != len(y) or len(y) != len(yerr):
        raise ValueError("dimension mismatch")

    return log_likelihood(
        log_sigma, len(arpars), <double*>arpars.data,
        len(mapars), <double*>mapars.data,
        len(x), <double*>x.data, <double*>y.data, <double*>yerr.data
    )


def carma_psd(
    double log_sigma,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] arpars,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] mapars,
    f
):
    check_pars(arpars, mapars)
    cdef np.ndarray[DTYPE_t] f_array = np.atleast_1d(f).ravel()
    cdef np.ndarray[DTYPE_t] values = np.empty_like(f_array)
    psd(log_sigma, len(arpars), <double*>arpars.data,
        len(mapars), <double*>mapars.data,
        f_array.size, <double*>f_array.data, <double*>values.data)
    try:
        return values.reshape(f.shape)
    except AttributeError:
        return float(values)


def carma_covariance(
    double log_sigma,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] arpars,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] mapars,
    tau
):
    check_pars(arpars, mapars)

    cdef np.ndarray[DTYPE_t] tau_array = np.abs(np.atleast_1d(tau).ravel())
    cdef np.ndarray[DTYPE_t] values = np.empty_like(tau_array)
    covariance(log_sigma, len(arpars), <double*>arpars.data,
               len(mapars), <double*>mapars.data,
               tau_array.size, <double*>tau_array.data, <double*>values.data)
    try:
        return values.reshape(tau.shape)
    except AttributeError:
        return float(values)
