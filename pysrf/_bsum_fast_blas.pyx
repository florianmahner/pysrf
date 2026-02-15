# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# BLAS-accelerated BSUM for symmetric NMF.
#
# Uses scipy's cython_blas for hot inner loops:
# - dgemv for per-row mw_i = M[i,:] @ W (95% of cost)
# - ddot for W[i,:] · WtW[j,:] dot products
# - daxpy for WtW rank-1 updates
# Plus raw C pointers for scalar operations.

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cbrt, fabs
from scipy.linalg.cython_blas cimport dgemv, ddot, daxpy

np.import_array()

ctypedef np.float64_t DTYPE_t


cpdef double _quartic_root(double a, double b, double c, double d) nogil:
    cdef double bb  = b * b
    cdef double a2  = a * a
    cdef double p   = (3.0 * a * c - bb) / (3.0 * a2)
    cdef double q   = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)
    cdef double root, delta, tmp

    if c > bb / (3.0 * a):
        delta = sqrt(q * q * 0.25 + p * p * p / 27.0)
        root  = cbrt(q * 0.5 - delta) + cbrt(q * 0.5 + delta)
    else:
        tmp   = b * bb / (27.0 * a2 * a) - d / a
        root  = cbrt(tmp)

    return 0.0 if root < 0.0 else root


cpdef np.ndarray[DTYPE_t, ndim=2] update_w(double[:, ::1] m,
                                            double[:, ::1] w0,
                                            int max_iter=100,
                                            double tol=1e-6):
    cdef int n = w0.shape[0]
    cdef int r = w0.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)
    cdef np.ndarray[DTYPE_t, ndim=1] mw_i_buf = np.empty(r)

    # Raw C pointers — all arrays are C-contiguous
    cdef double* w_p = <double*>w.data
    cdef double* wtw_p = <double*>wtw.data
    cdef double* diag_p = <double*>diag.data
    cdef double* mwi_p = <double*>mw_i_buf.data
    cdef double* m_p = &m[0, 0]

    # BLAS parameters
    cdef char trans_t = b'T'
    cdef char trans_n = b'N'
    cdef int blas_n = n
    cdef int blas_r = r
    cdef int inc_1 = 1
    cdef double alpha_1 = 1.0
    cdef double beta_0 = 0.0

    cdef int it, i, j, k
    cdef int ir, jr, in_off
    cdef double max_delta, old_val, b_coef, c_coef, d_coef
    cdef double new_val, delta_val, dot_val

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            ir = i * r
            in_off = i * n

            # mw_i = M[i,:] @ W = W^T @ M[i,:] via BLAS dgemv
            # C row-major W(n,r) = Fortran col-major A(r,n) with lda=r, where A = W^T
            # y = A @ x: trans='N', m=r, n=n, lda=r
            dgemv(&trans_n, &blas_r, &blas_n, &alpha_1,
                  w_p, &blas_r,
                  &m_p[in_off], &inc_1,
                  &beta_0, mwi_p, &inc_1)

            for j in range(r):
                jr = j * r
                old_val = w_p[ir + j]

                b_coef = 12.0 * old_val
                c_coef = 4.0 * ((diag_p[i] - m_p[in_off + i]) + wtw_p[jr + j] + old_val * old_val)

                # W[i,:] · WtW[j,:] via BLAS ddot (both contiguous rows of length r)
                dot_val = ddot(&blas_r, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                d_coef = 4.0 * dot_val - 4.0 * mwi_p[j]
                new_val = _quartic_root(4.0, b_coef, c_coef, d_coef)
                delta_val = new_val - old_val

                if fabs(delta_val) > max_delta:
                    max_delta = fabs(delta_val)

                diag_p[i] += new_val * new_val - old_val * old_val

                # WtW[j,:] += delta * W[i,:] via BLAS daxpy
                daxpy(&blas_r, &delta_val, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                # WtW[:,j] += delta * W[i,:] (strided column update)
                for k in range(r):
                    wtw_p[k * r + j] += delta_val * w_p[ir + k]

                wtw_p[jr + j] += delta_val * delta_val

                w_p[ir + j] = new_val

        if max_delta < tol and tol > 0.0:
            break

    return w
