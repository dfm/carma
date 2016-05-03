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
    cdef double compute_log_likelihood(
        double sigma, unsigned p, complex[double]* ar, unsigned q, complex[double]* ma,
        unsigned n, double* t, double* y, double* yerr
    )

def carma_log_likelihood(
    double sigma,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] arroots,
    np.ndarray[CDTYPE_t, ndim=1, mode='c'] maroots,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] yerr
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

    if len(x) != len(y) or len(y) != len(yerr):
        raise ValueError("dimension mismatch")

    return compute_log_likelihood(
        sigma, len(arroots), <complex[double]*>arroots.data,
        len(maroots), <complex[double]*>maroots.data,
        len(x), <double*>x.data, <double*>y.data, <double*>yerr.data
    )
