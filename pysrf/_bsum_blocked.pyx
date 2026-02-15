# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# Blocked BSUM for symmetric NMF with exact intra-block corrections.
#
# Replaces per-row dgemv (BLAS-2, memory-bound) with:
#   1. dsymm to precompute MW = M @ W (BLAS-3, compute-bound)
#   2. dgemm for inter-block corrections (BLAS-3)
#   3. Cheap scalar-vector corrections within each block for exact equivalence
#
# Mathematically identical to _bsum_fast_blas for ANY block_size.

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cbrt, fabs
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport ddot, daxpy, dsymm, dgemm

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
                                            double tol=1e-6,
                                            int block_size=0):
    cdef int n = w0.shape[0]
    cdef int r = w0.shape[1]

    if block_size <= 0:
        block_size = min(50, max(1, n // 10))

    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)

    # MW buffer: precomputed M @ W, corrected per block
    cdef np.ndarray[DTYPE_t, ndim=2] mw = np.empty((n, r), dtype=np.float64)

    # Per-row delta buffer: stores delta_W for each row in current block
    cdef np.ndarray[DTYPE_t, ndim=2] delta_w = np.empty((block_size, r), dtype=np.float64)

    # Snapshot of W[i,:] before processing row i (for computing per-row delta)
    cdef np.ndarray[DTYPE_t, ndim=1] w_old_row = np.empty(r, dtype=np.float64)

    # Raw C pointers
    cdef double* w_p = <double*>w.data
    cdef double* wtw_p = <double*>wtw.data
    cdef double* diag_p = <double*>diag.data
    cdef double* mw_p = <double*>mw.data
    cdef double* m_p = &m[0, 0]
    cdef double* dw_p = <double*>delta_w.data
    cdef double* w_old_p = <double*>w_old_row.data

    # BLAS parameters
    cdef char side_r = b'R'
    cdef char uplo_l = b'L'
    cdef char trans_n = b'N'
    cdef int inc_1 = 1
    cdef double alpha_1 = 1.0
    cdef double beta_0 = 0.0
    cdef double beta_1 = 1.0

    # BLAS dimension variables
    cdef int blas_n = n
    cdef int blas_r = r

    cdef int it, i, j, k, bi, prev
    cdef int ir, jr, in_off
    cdef int block_start, block_end, block_len
    cdef int remaining, blas_remaining, blas_block_len
    cdef double max_delta, old_val, b_coef, c_coef, d_coef
    cdef double new_val, delta_val, dot_val
    cdef double m_ik

    for it in range(max_iter):
        max_delta = 0.0

        # Precompute MW = M @ W via dsymm (BLAS-3)
        # C row-major (n,r) maps to Fortran col-major (r,n)
        # MW = M @ W => in Fortran: C(r,n) = W(r,n) * M(n,n) => side='R'
        dsymm(&side_r, &uplo_l, &blas_r, &blas_n, &alpha_1,
              m_p, &blas_n,
              w_p, &blas_r,
              &beta_0, mw_p, &blas_r)

        block_start = 0
        while block_start < n:
            block_end = block_start + block_size
            if block_end > n:
                block_end = n
            block_len = block_end - block_start

            # Inter-block correction: apply accumulated deltas from all previous blocks
            # MW[block_start:block_end,:] += M[block_start:block_end, 0:block_start] @ DeltaW_accum[0:block_start,:]
            # But we don't store all accumulated deltas. Instead, the inter-block dgemm
            # at the end of each block already corrects all REMAINING rows.
            # So MW[block_start:block_end,:] is already corrected for previous blocks.

            for bi in range(block_len):
                i = block_start + bi
                ir = i * r
                in_off = i * n

                # Intra-block correction: correct MW[i,:] for changes from rows
                # block_start..block_start+bi-1 (earlier rows in this block)
                # MW[i,:] += sum_{k=0..bi-1} M[i, block_start+k] * delta_w[k,:]
                # Each correction is: MW[i,:] += scalar * vector (length r)
                for prev in range(bi):
                    m_ik = m_p[in_off + block_start + prev]
                    # MW[i,:] += M[i, block_start+prev] * delta_w[prev,:]
                    daxpy(&blas_r, &m_ik, &dw_p[prev * r], &inc_1, &mw_p[ir], &inc_1)

                # Save W[i,:] before j-loop modifies it
                memcpy(w_old_p, &w_p[ir], r * sizeof(double))

                for j in range(r):
                    jr = j * r
                    old_val = w_p[ir + j]

                    b_coef = 12.0 * old_val
                    c_coef = 4.0 * ((diag_p[i] - m_p[in_off + i]) + wtw_p[jr + j] + old_val * old_val)

                    dot_val = ddot(&blas_r, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                    d_coef = 4.0 * dot_val - 4.0 * mw_p[ir + j]
                    new_val = _quartic_root(4.0, b_coef, c_coef, d_coef)
                    delta_val = new_val - old_val

                    if fabs(delta_val) > max_delta:
                        max_delta = fabs(delta_val)

                    diag_p[i] += new_val * new_val - old_val * old_val

                    daxpy(&blas_r, &delta_val, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                    for k in range(r):
                        wtw_p[k * r + j] += delta_val * w_p[ir + k]

                    wtw_p[jr + j] += delta_val * delta_val
                    w_p[ir + j] = new_val

                # Store delta for this row: delta_w[bi,:] = W_new[i,:] - W_old[i,:]
                for j in range(r):
                    dw_p[bi * r + j] = w_p[ir + j] - w_old_p[j]

            # Inter-block correction: correct MW for all remaining rows using dgemm
            # MW[block_end:,:] += M[block_end:, block_start:block_end] @ DeltaW[0:block_len,:]
            # Fortran layout:
            #   DW(block_len, r) in C = DW^F(r, block_len) with lda=r
            #   M_sub(remaining, block_len) in C = M_sub^F(block_len, remaining) with lda=n
            #   MW_sub(remaining, r) in C = MW_sub^F(r, remaining) with lda=r
            # dgemm('N','N', r, remaining, block_len, 1.0, DW, r, M_sub, n, 1.0, MW_sub, r)
            remaining = n - block_end
            if remaining > 0:
                blas_remaining = remaining
                blas_block_len = block_len
                dgemm(&trans_n, &trans_n, &blas_r, &blas_remaining, &blas_block_len,
                      &alpha_1,
                      dw_p, &blas_r,
                      &m_p[block_end * n + block_start], &blas_n,
                      &beta_1,
                      &mw_p[block_end * r], &blas_r)

            block_start = block_end

        if max_delta < tol and tol > 0.0:
            break

    return w
