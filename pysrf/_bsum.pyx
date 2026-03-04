# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# Author: Florian P. Mahner
# Date: 2025-10-08
# Description: Cython implementation of the block successive upper bound minimization algorithm for symmetric nonnegative matrix factorization.
# See Shi et al. (2016) for more details.
# Reference: Shi et al. (2016) "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization".
# Link: https://arxiv.org/abs/1608.02649
# License: MIT

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from pysrf._quartic cimport _quartic_root

np.import_array()

ctypedef np.float64_t DTYPE_t


cdef inline double _dot_product(double[:] a, double[:] b) nogil:
    cdef int n = a.shape[0]
    cdef int i
    cdef double result = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
    return result


cpdef np.ndarray[DTYPE_t, ndim=2] update_w(double[:, ::1] m,
                                                double[:, ::1] w0,
                                                int max_iter=100,
                                                double tol=1e-6):
    cdef int n = w0.shape[0]
    cdef int r = w0.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)
    
    cdef double[:, :] w_view = w
    cdef double[:, :] m_view = m
    cdef double[:, :] wtw_view = wtw
    cdef double[:] diag_view = diag
    
    cdef double a = 4.0
    cdef int it, i, j, k
    cdef double max_delta, old, b, c, d, new, delta, update_val
    
    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = w_view[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag_view[i] - m_view[i, i]) + wtw_view[j, j] + old * old)
                
                d = 4.0 * _dot_product(w_view[i, :], wtw_view[:, j]) - 4.0 * _dot_product(m_view[i, :], w_view[:, j])
                new = _quartic_root(a, b, c, d)
                delta = new - old
                
                if fabs(delta) > max_delta:
                    max_delta = fabs(delta)
                
                diag_view[i] += new * new - old * old

                for k in range(r):
                    update_val = delta * w_view[i, k]
                    wtw_view[j, k] += update_val
                    wtw_view[k, j] += update_val
                
                wtw_view[j, j] += delta * delta
                
                w_view[i, j] = new
        
        if max_delta < tol and tol > 0.0:
            break
    
    return w

