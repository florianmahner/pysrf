# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython implementation of the vector-wise BSUM algorithm for symmetric NMF.

Based on Algorithm 2 from:
Shi et al. (2016) "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization"
https://arxiv.org/abs/1608.02649

This implementation updates entire rows at once, which:
1. Uses BLAS for the O(n²r) M @ W computation (much faster than element-wise)
2. Has better cache locality
3. Can converge faster due to larger block updates
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, cbrt

np.import_array()

ctypedef np.float64_t DTYPE_t


cdef inline double _max_abs_double(double[:] arr) noexcept nogil:
    """Find maximum absolute value in array."""
    cdef int n = arr.shape[0]
    cdef int i
    cdef double max_val = 0.0
    cdef double abs_val

    for i in range(n):
        abs_val = fabs(arr[i])
        if abs_val > max_val:
            max_val = abs_val
    return max_val


cdef inline double _norm(double[:] arr) noexcept nogil:
    """Compute L2 norm of array."""
    cdef int n = arr.shape[0]
    cdef int i
    cdef double sum_sq = 0.0

    for i in range(n):
        sum_sq += arr[i] * arr[i]
    return sqrt(sum_sq)


cdef inline void _solve_row_update(double[:] bi, double sQi, double[:] result) noexcept nogil:
    """
    Solve the row update subproblem according to Equation (22).

    Solution: x = t * [bi]+ / ||[bi]+||  if ||[bi]+|| > 0, else x = 0

    where:
        t = cbrt(||[bi]+||/2 + sqrt(Δ)) + cbrt(||[bi]+||/2 - sqrt(Δ))
        Δ = ||[bi]+||^2 / 4 + sQi^3 / 27
    """
    cdef int r = bi.shape[0]
    cdef int j
    cdef double bi_plus_norm_sq = 0.0
    cdef double bi_plus_norm, delta, sqrt_delta, half_norm, t, scale

    # Compute ||[bi]+||
    for j in range(r):
        if bi[j] > 0:
            bi_plus_norm_sq += bi[j] * bi[j]

    bi_plus_norm = sqrt(bi_plus_norm_sq)

    if bi_plus_norm <= 1e-16:
        for j in range(r):
            result[j] = 0.0
        return

    # Δ = ||[bi]+||^2 / 4 + sQi^3 / 27
    delta = bi_plus_norm_sq / 4.0 + (sQi * sQi * sQi) / 27.0

    # t = cbrt(||[bi]+||/2 + sqrt(Δ)) + cbrt(||[bi]+||/2 - sqrt(Δ))
    sqrt_delta = sqrt(delta)
    half_norm = bi_plus_norm / 2.0

    t = cbrt(half_norm + sqrt_delta) + cbrt(half_norm - sqrt_delta)

    # x = t * [bi]+ / ||[bi]+||
    scale = t / bi_plus_norm
    for j in range(r):
        if bi[j] > 0:
            result[j] = scale * bi[j]
        else:
            result[j] = 0.0


cpdef np.ndarray[DTYPE_t, ndim=2] update_w_vbsum(
    double[:, ::1] m,
    double[:, ::1] w0,
    int max_iter=100,
    double tol=1e-6,
    int imax=5
):
    """
    Vector-wise BSUM algorithm for symmetric NMF.

    Updates entire rows of W at once using the vBSUM algorithm from
    Shi et al. (2016), Algorithm 2.

    Parameters
    ----------
    m : ndarray of shape (n, n)
        Target symmetric matrix to factorize (must be C-contiguous)
    w0 : ndarray of shape (n, r)
        Initial factor matrix (must be C-contiguous)
    max_iter : int, default=100
        Maximum number of outer iterations
    tol : float, default=1e-6
        Convergence tolerance (max change in any row)
    imax : int, default=5
        Number of inner iterations per row update

    Returns
    -------
    w : ndarray of shape (n, r)
        Optimized factor matrix

    Notes
    -----
    Complexity per outer iteration: O(n²r + nr²·imax)
    - O(n²r) for computing W^T @ M using BLAS
    - O(nr²·imax) for the inner row updates

    For large n and small r, this is faster than sBSUM which has O(n²r)
    per iteration but with slow element-wise operations.
    """
    cdef int n = w0.shape[0]
    cdef int r = w0.shape[1]

    # Copy initial W
    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, dtype=np.float64, copy=True, order='C')
    cdef double[:, :] w_view = w
    cdef double[:, :] m_view = m

    # Allocate working arrays
    cdef np.ndarray[DTYPE_t, ndim=2] wtw_np = np.dot(w.T, w)  # r x r
    cdef double[:, :] wtw = wtw_np

    cdef np.ndarray[DTYPE_t, ndim=2] Pi_np = np.zeros((r, r), dtype=np.float64)
    cdef double[:, :] Pi = Pi_np

    cdef np.ndarray[DTYPE_t, ndim=1] qi_np = np.zeros(r, dtype=np.float64)
    cdef double[:] qi = qi_np

    cdef np.ndarray[DTYPE_t, ndim=1] bi_np = np.zeros(r, dtype=np.float64)
    cdef double[:] bi = bi_np

    cdef np.ndarray[DTYPE_t, ndim=1] xi_np = np.zeros(r, dtype=np.float64)
    cdef double[:] xi = xi_np

    cdef np.ndarray[DTYPE_t, ndim=1] old_row_np = np.zeros(r, dtype=np.float64)
    cdef double[:] old_row = old_row_np

    cdef np.ndarray[DTYPE_t, ndim=1] eigs_np = np.zeros(r, dtype=np.float64)

    # Variables for iterations
    cdef int it, i, j, k, inner, ii
    cdef double max_delta, row_delta, sQi, Mii
    cdef double temp, max_abs_eig

    # Store M @ X (like MATLAB) for incremental updates
    # Compute once before the loop, then update incrementally
    cdef np.ndarray[DTYPE_t, ndim=2] MX_np = np.dot(m, w)
    cdef double[:, :] MX = MX_np

    cdef np.ndarray[DTYPE_t, ndim=1] delta_row_np = np.zeros(r, dtype=np.float64)
    cdef double[:] delta_row = delta_row_np

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            Mii = m_view[i, i]

            # Save old row for convergence check
            for j in range(r):
                old_row[j] = w_view[i, j]

            # Step 3: Pi = (X^T X) - X[i,:]^T @ X[i,:]
            for j in range(r):
                for k in range(r):
                    Pi[j, k] = wtw[j, k] - w_view[i, j] * w_view[i, k]

            # Step 4: qi = (M @ X)[i,:] - M[i,i] * X[i,:]
            # For symmetric M: (M @ X)[i,:] = X^T @ M[:,i]
            for j in range(r):
                qi[j] = MX[i, j] - Mii * w_view[i, j]

            # Compute Lipschitz constant sA = max eigenvalue of Pi
            # This matches the MATLAB implementation exactly
            # Pi is PSD so all eigenvalues are non-negative
            eigs_np = np.linalg.eigvalsh(Pi_np)
            sQi = eigs_np[r-1]  # eigvalsh returns sorted ascending, so last is max
            if sQi < 1e-10:
                sQi = 1e-10

            # Initialize xi with current row
            for j in range(r):
                xi[j] = w_view[i, j]

            # Steps 5-8: Inner iterations
            for inner in range(imax):
                # Step 6: bi = qi + (sQi + Mii) * xi - Pi @ xi
                for j in range(r):
                    temp = 0.0
                    for k in range(r):
                        temp += Pi[j, k] * xi[k]
                    bi[j] = qi[j] + (sQi + Mii) * xi[j] - temp

                # Step 7: Update xi according to (22)
                _solve_row_update(bi, sQi, xi)

            # Update w[i, :]
            for j in range(r):
                w_view[i, j] = xi[j]

            # Step 9: Update wtw = Pi + xi^T @ xi
            for j in range(r):
                for k in range(r):
                    wtw[j, k] = Pi[j, k] + xi[j] * xi[k]

            # Incremental MX update (matches MATLAB exactly):
            # MX = MX - M[:,i] * (X_old[i,:] - xi)^T
            # This is an outer product: M[:,i] is (n,), (old_row - xi) is (r,)
            for ii in range(n):
                for j in range(r):
                    MX[ii, j] = MX[ii, j] - m_view[ii, i] * (old_row[j] - xi[j])

            # Track convergence
            row_delta = 0.0
            for j in range(r):
                temp = fabs(xi[j] - old_row[j])
                if temp > row_delta:
                    row_delta = temp
            if row_delta > max_delta:
                max_delta = row_delta

        # Check convergence
        if max_delta < tol and tol > 0.0:
            break

    return w
