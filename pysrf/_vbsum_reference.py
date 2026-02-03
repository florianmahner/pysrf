"""
Reference Python implementation of vBSUM (Vector-wise Block Successive Upper Bound Minimization).

Based on Algorithm 2 from:
Shi et al. (2016) "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization"
https://arxiv.org/abs/1608.02649

This is a reference implementation for testing correctness before Cython optimization.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def _compute_lipschitz_constant(Pi: np.ndarray, Mii: float) -> float:
    """
    Compute Lipschitz constant sQi for the quadratic term x^T @ Qi @ x.

    sQi is the spectral norm of Qi = Pi - Mii * I.
    This is max(|eigenvalues(Qi)|) = max(|eigenvalues(Pi) - Mii|).

    Parameters
    ----------
    Pi : ndarray of shape (r, r)
        Symmetric positive semidefinite matrix (X without row i)^T @ (X without row i)
    Mii : float
        Diagonal element M[i,i]

    Returns
    -------
    sQi : float
        Lipschitz constant (always positive)
    """
    # Compute eigenvalues of Pi (symmetric, so real eigenvalues)
    eigs_Pi = np.linalg.eigvalsh(Pi)

    # Eigenvalues of Qi = Pi - Mii*I are eigs_Pi - Mii
    eigs_Qi = eigs_Pi - Mii

    # Lipschitz constant = spectral norm = max absolute eigenvalue
    sQi = np.max(np.abs(eigs_Qi))

    # Ensure positive
    return max(sQi, 1e-10)


def _solve_row_update(bi: np.ndarray, sQi: float) -> np.ndarray:
    """
    Solve the row update subproblem according to Equation (22) in the paper.

    The solution is:
        x = t * [bi]+ / ||[bi]+||  if ||[bi]+|| > 0
        x = 0                       otherwise

    where:
        t = cbrt(||[bi]+||/2 - sqrt(Δ)) + cbrt(||[bi]+||/2 + sqrt(Δ))
        Δ = ||[bi]+||^2 / 4 + sQi^3 / 27

    Parameters
    ----------
    bi : ndarray of shape (r,)
        The bi vector from the algorithm
    sQi : float
        Lipschitz constant (must be positive)

    Returns
    -------
    x : ndarray of shape (r,)
        The updated row
    """
    bi_plus = np.maximum(bi, 0.0)
    bi_plus_norm = norm(bi_plus)

    if bi_plus_norm <= 1e-16:
        return np.zeros_like(bi)

    # Δ = ||[bi]+||^2 / 4 + sQi^3 / 27
    delta = (bi_plus_norm ** 2) / 4.0 + (sQi ** 3) / 27.0

    # t = cbrt(||[bi]+||/2 - sqrt(Δ)) + cbrt(||[bi]+||/2 + sqrt(Δ))
    sqrt_delta = np.sqrt(delta)
    half_norm = bi_plus_norm / 2.0

    t = np.cbrt(half_norm + sqrt_delta) + np.cbrt(half_norm - sqrt_delta)

    # x = t * [bi]+ / ||[bi]+||
    x = t * bi_plus / bi_plus_norm

    return x


def update_w_vbsum_python(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    imax: int = 1,
) -> np.ndarray:
    """
    Vector-wise BSUM algorithm for symmetric NMF (Python reference implementation).

    Updates entire rows of W at once, which can converge faster than element-wise updates.

    Parameters
    ----------
    m : ndarray of shape (n, n)
        Target symmetric matrix to factorize
    w0 : ndarray of shape (n, r)
        Initial factor matrix
    max_iter : int, default=100
        Maximum number of outer iterations
    tol : float, default=1e-6
        Convergence tolerance (max change in any element)
    imax : int, default=1
        Number of inner iterations per row update

    Returns
    -------
    w : ndarray of shape (n, r)
        Optimized factor matrix

    References
    ----------
    Algorithm 2 in Shi et al. (2016), Table II
    """
    w = w0.copy()
    n, r = w.shape

    # Initialize X^T @ X (r x r matrix)
    wtw = w.T @ w

    # Initialize MX = M @ X once, then update incrementally (matches MATLAB)
    MX = m @ w  # shape: (n, r)

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            # Store old row for convergence check
            old_row = w[i, :].copy()

            # Step 3: Pi = (X^T @ X) - X[i,:]^T @ X[i,:]
            # This removes row i's contribution from wtw
            Pi = wtw - np.outer(w[i, :], w[i, :])

            # Step 4: qi = (M @ X)[i,:] - M[i,i] * X[i,:] (matches MATLAB: vtmp)
            qi = MX[i, :] - m[i, i] * w[i, :]

            # Compute Lipschitz constant sA = max eigenvalue of Pi (matches MATLAB)
            eigs = np.linalg.eigvalsh(Pi)
            sQi = eigs[-1]  # max eigenvalue (eigvalsh returns sorted ascending)
            sQi = max(sQi, 1e-10)

            # Steps 5-8: Inner iterations
            xi = w[i, :].copy()
            for _ in range(imax):
                # Step 6: bi = qi + (sQi + M[i,i]) * X[i,:]^T - Pi @ X[i,:]^T
                bi = qi + (sQi + m[i, i]) * xi - Pi @ xi

                # Step 7: Update Xi: according to (22)
                xi = _solve_row_update(bi, sQi)

            # Update the row
            w[i, :] = xi

            # Step 9: Update X^T @ X = Pi + X[i,:]^T @ X[i,:]
            wtw = Pi + np.outer(xi, xi)

            # Incremental MX update (matches MATLAB exactly):
            # MX = MX - M[:,i] @ (X_old[i,:] - xi)^T
            MX = MX - np.outer(m[:, i], old_row - xi)

            # Track convergence
            row_delta = np.max(np.abs(xi - old_row))
            max_delta = max(max_delta, row_delta)

        # Check convergence
        if max_delta < tol and tol > 0.0:
            break

    return w


def update_w_sbsum_python(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Scalar-wise BSUM algorithm for symmetric NMF (Python reference implementation).

    This is Algorithm 1 from the paper - updates one element at a time.
    Reimplemented here for direct comparison with vBSUM.

    Parameters
    ----------
    m : ndarray of shape (n, n)
        Target symmetric matrix to factorize
    w0 : ndarray of shape (n, r)
        Initial factor matrix
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    w : ndarray of shape (n, r)
        Optimized factor matrix
    """
    w = w0.copy()
    n, r = w.shape

    # Initialize X^T @ X and diagonal of X @ X^T
    wtw = w.T @ w  # r x r
    diag_wwt = np.einsum('ij,ij->i', w, w)  # diagonal of w @ w.T

    a = 4.0

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            for j in range(r):
                old = w[i, j]

                # Compute coefficients (Equations 6-9)
                b = 12.0 * old
                c = 4.0 * ((diag_wwt[i] - m[i, i]) + wtw[j, j] + old * old)

                # d = 4 * (X @ (X^T @ X) - M @ X)[i,j]
                #   = 4 * X[i,:] @ (X^T @ X)[:,j] - 4 * M[i,:] @ X[:,j]
                d = 4.0 * np.dot(w[i, :], wtw[:, j]) - 4.0 * np.dot(m[i, :], w[:, j])

                # Solve the quartic minimization (Lemma 2.2)
                new = _solve_quartic(a, b, c, d)

                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # Update diag_wwt[i]
                diag_wwt[i] += new * new - old * old

                # Update wtw (Steps 8-11 in Algorithm 1)
                update_vec = delta * w[i, :]
                wtw[j, :] += update_vec
                wtw[:, j] += update_vec
                wtw[j, j] += delta * delta

                w[i, j] = new

        if max_delta < tol and tol > 0.0:
            break

    return w


def _solve_quartic(a: float, b: float, c: float, d: float) -> float:
    """
    Solve the quartic minimization problem from Lemma 2.2.

    Find x >= 0 minimizing g(x) = a/4 * x^4 + b/3 * x^3 + c/2 * x^2 + d * x

    Parameters
    ----------
    a, b, c, d : float
        Polynomial coefficients (a is always 4)

    Returns
    -------
    x : float
        Non-negative minimizer
    """
    bb = b * b
    a2 = a * a

    # p and q for Cardano's formula
    p = (3.0 * a * c - bb) / (3.0 * a2)
    q = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)

    # Check condition c > b^2 / (3a)
    if c > bb / (3.0 * a):
        # Use Cardano's formula
        delta = np.sqrt(q * q * 0.25 + p * p * p / 27.0)
        root = np.cbrt(q * 0.5 - delta) + np.cbrt(q * 0.5 + delta)
    else:
        # Simplified case
        tmp = b * bb / (27.0 * a2 * a) - d / a
        root = np.cbrt(tmp)

    return max(0.0, root)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n, r = 50, 5

    # Generate test matrix
    W_true = np.random.rand(n, r)
    M = W_true @ W_true.T
    M = (M + M.T) / 2

    # Initial guess
    W0 = np.random.rand(n, r)

    # Run both algorithms
    W_sbsum = update_w_sbsum_python(M, W0.copy(), max_iter=100, tol=0.0)
    W_vbsum = update_w_vbsum_python(M, W0.copy(), max_iter=100, tol=0.0, imax=1)

    # Compare reconstructions
    rec_sbsum = W_sbsum @ W_sbsum.T
    rec_vbsum = W_vbsum @ W_vbsum.T

    err_sbsum = np.linalg.norm(M - rec_sbsum, 'fro')
    err_vbsum = np.linalg.norm(M - rec_vbsum, 'fro')

    print(f"sBSUM reconstruction error: {err_sbsum:.6f}")
    print(f"vBSUM reconstruction error: {err_vbsum:.6f}")
    print(f"Difference in W: {np.max(np.abs(W_sbsum - W_vbsum)):.6e}")
