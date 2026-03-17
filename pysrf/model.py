"""
Symmetric Non-negative Matrix Factorization via BSUM/ADMM.

SRF uses Block Successive Upper-bound Minimization (BSUM) for the W-subproblem
and Alternating Direction Method of Multipliers (ADMM) for the missing-data path.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import trange
from sklearn.utils.validation import (
    check_is_fitted,
    check_symmetric,
    validate_data,
)
from sklearn.utils._param_validation import Interval, StrOptions, Integral, Real


logger = logging.getLogger(__name__)


def _frobenius_residual(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Compute ||X - WW'||_F and ||WW'||_F without forming the n x n product.

    Parameters
    ----------
    x : ndarray of shape (n, n)
        Symmetric input matrix
    w : ndarray of shape (n, r)
        Factor matrix

    Returns
    -------
    residual_norm : float
        Frobenius norm of the reconstruction error
    xhat_norm : float
        Frobenius norm of the approximation WW'
    """
    xw = x @ w
    x_norm_sq = np.sum(x * x)
    trace_xwwt = np.sum(xw * w)
    wtw = w.T @ w
    wwt_norm_sq = np.sum(wtw * wtw)
    residual_sq = max(0.0, x_norm_sq - 2.0 * trace_xwwt + wwt_norm_sq)
    return np.sqrt(residual_sq), np.sqrt(wwt_norm_sq)


def _solve_quartic_minimization(a: float, b: float, c: float, d: float) -> float:
    """Find x >= 0 minimizing g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d*x.

    Key subroutine in the BSUM algorithm for updating individual
    elements of the factor matrix W.

    Parameters
    ----------
    a, b, c, d : float
        Polynomial coefficients

    Returns
    -------
    root : float
        Non-negative minimizer of g(x)

    References
    ----------
    Shi et al. (2016), "Inexact Block Coordinate Descent Methods For
    Symmetric Nonnegative Matrix Factorization", Equation (11)
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    bb = b * b
    a2 = a * a
    p = (3.0 * a * c - bb) / (3.0 * a2)

    q = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)

    if c > bb / (3.0 * a):
        delta = math.sqrt(q * q * 0.25 + p * p * p / 27.0)
        root = math.cbrt(q * 0.5 - delta) + math.cbrt(q * 0.5 + delta)
    else:
        tmp = b * bb / (27.0 * a2 * a) - d / a
        root = math.cbrt(tmp)

    return max(0.0, root)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product via scalar accumulation, matching Cython summation order.

    Parameters
    ----------
    a, b : ndarray of shape (n,)
        Input vectors

    Returns
    -------
    result : float
        Dot product of a and b
    """
    return sum(x * y for x, y in zip(a, b))


def _is_nan_marker(missing_values: float | None) -> bool:
    """Check if missing_values represents NaN.

    Needed because sklearn clone() can break ``is np.nan`` identity checks.

    Parameters
    ----------
    missing_values : float or None
        The sentinel value to test

    Returns
    -------
    is_nan : bool
        True if missing_values is NaN
    """
    return missing_values is np.nan or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    )


def _get_missing_mask(x: np.ndarray, missing_values: float | None) -> np.ndarray:
    """Boolean mask where True marks missing entries in x.

    Parameters
    ----------
    x : ndarray of shape (n, n)
        Input matrix
    missing_values : float or None
        Sentinel value for missing data. If None or NaN, uses ``np.isnan``.

    Returns
    -------
    mask : ndarray of shape (n, n)
        Boolean array, True where x has missing values
    """
    if _is_nan_marker(missing_values) or missing_values is None:
        return np.isnan(x)
    else:
        return x == missing_values


def _initialize_w(
    x: np.ndarray,
    rank: int,
    method: str = "random_sqrt",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Initialize factor matrix W for symmetric NMF.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_samples)
        Symmetric input matrix (used for scaling)
    rank : int
        Number of components (columns in W)
    method : {'random', 'random_sqrt'}, default='random_sqrt'
        Initialization strategy:

        - 'random' : Uniform [0, 0.1], simple baseline
        - 'random_sqrt' : Uniform scaled by sqrt(X.mean() / rank)
    random_state : int, RandomState or None
        Controls random number generation

    Returns
    -------
    w : ndarray of shape (n_samples, n_components)
        Initialized non-negative factor matrix
    """
    rng = np.random.RandomState(random_state)
    n_samples = x.shape[0]

    if method == "random":
        w = 0.1 * rng.rand(n_samples, rank)
    elif method == "random_sqrt":
        avg = np.sqrt(x.mean() / rank)
        w = rng.rand(n_samples, rank) * avg
    else:
        raise ValueError(
            f"Invalid initialization method '{method}'. "
            f"Choose from: 'random', 'random_sqrt'"
        )

    return w


def _validate_bounds(val) -> None:
    """Raise ValueError if val is not a valid (lower, upper) bounds tuple.

    Parameters
    ----------
    val : tuple of (float, float) or None
        Bounds to validate. None means no bounds.
    """
    if val is None:
        return
    if not (isinstance(val, tuple) and len(val) == 2):
        raise ValueError("bounds must be a tuple (lower, upper) of floats or None")
    lo, up = val
    # allow None for either end
    if lo is None or up is None:
        return
    try:
        lo_f = float(lo)
        up_f = float(up)
    except (TypeError, ValueError):
        raise ValueError("bounds must be numeric or None")
    if lo_f > up_f:
        raise ValueError("lower bound cannot be greater than upper bound")


def update_w(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Block successive upper-bound minimization (BSUM) for SymNMF.

    Solves the W-subproblem by minimizing ||M - WW'||^2_F element-wise
    via quartic surrogate functions. See TABLE 1 in Shi et al. (2016),
    "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative
    Matrix Factorization".

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
    xtx = w.T @ w
    diag = np.einsum("ij,ij->i", w, w)
    a = 4.0

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = w[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4.0 * _dot(w[i], xtx[:, j]) - 4.0 * _dot(m[i], w[:, j])
                new = _solve_quartic_minimization(a, b, c, d)

                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * w[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                w[i, j] = new

        if max_delta < tol and tol > 0.0:
            break

    return w


def _resolve_solver() -> tuple[callable, str]:
    """Return the best available BSUM solver and its source label.

    Tries the Cython BLAS-blocked implementation first; falls back to the
    pure-Python solver with a warning.

    Returns
    -------
    solver : callable
        The update_w function to use
    source : str
        ``'cython'`` or ``'python'``
    """
    try:
        from ._bsum import update_w_blas_blocked

        return update_w_blas_blocked, "cython"
    except ImportError:
        import warnings

        warnings.warn(
            "Cython implementation not available. Using Python implementation "
            "which is significantly slower. Compile Cython to improve performance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return update_w, "python"


_update_w_impl, _update_w_source = _resolve_solver()


def update_v_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """Update auxiliary variable V in the ADMM algorithm.

    For unobserved entries: v_ij = x_hat_ij - lambda_ij / rho.
    For observed entries: v_ij = (x_ij + rho * x_hat_ij - lambda_ij) / (1 + rho).
    The result is symmetrized and optionally clipped to [bound_min, bound_max].

    Parameters
    ----------
    observed_mask : ndarray of shape (n, n)
        Binary observation mask
    x : ndarray of shape (n, n)
        Original similarity matrix
    x_hat : ndarray of shape (n, n)
        Current estimate WW'
    lam : ndarray of shape (n, n)
        Lagrange multipliers
    rho : float
        Penalty parameter
    bound_min : float or None
        Lower bound constraint
    bound_max : float or None
        Upper bound constraint
    v : ndarray of shape (n, n)
        Auxiliary variable, updated in place
    """
    v[:] = lam
    v /= -rho
    v += x_hat

    v[observed_mask] = (
        x[observed_mask] + rho * x_hat[observed_mask] - lam[observed_mask]
    ) / (1.0 + rho)

    # since this optimization problem is linear we can do a projection step here to respect the bounds
    if bound_min is not None or bound_max is not None:
        np.clip(v, bound_min, bound_max, out=v)
    v += v.T
    v *= 0.5


def update_lambda_(
    lam: np.ndarray, v: np.ndarray, x_hat: np.ndarray, rho: float
) -> None:
    """Update Lagrange multipliers via dual ascent: lam += rho * (V - WW').

    Parameters
    ----------
    lam : ndarray of shape (n, n)
        Current Lagrange multipliers
    v : ndarray of shape (n, n)
        Auxiliary variable
    x_hat : ndarray of shape (n, n)
        Current matrix estimate WW'
    rho : float
        Penalty parameter
    """
    lam += rho * (v - x_hat)


class SRF(TransformerMixin, BaseEstimator):
    """Symmetric non-negative matrix factorization via ADMM.

    Factorizes a symmetric similarity matrix S into WW' where W >= 0.
    Handles missing entries and optional bound constraints on V.

    The objective is:
        min_{W>=0, V} ||M * (S - V)||^2_F + rho/2 * ||V - WW'||^2_F
    where M is the observation mask.

    Parameters
    ----------
    rank : int, default=10
        Number of factors (dimensionality of the latent space)
    rho : float, default=3.0
        SRF penalty parameter controlling constraint enforcement
    max_outer : int, default=10
        Maximum number of SRF outer iterations
    max_inner : int, default=30
        Maximum iterations for w-subproblem per outer iteration
    tol : float, default=1e-4
        Convergence tolerance for constraint violation
    verbose : int, default=0
        Whether to print optimization progress
    init : str, default='random_sqrt'
        Method for factor initialization ('random', 'random_sqrt')
    random_state : int or None, default=None
        Random seed for reproducible initialization
    missing_values : float or None, default=np.nan
        Values to be treated as missing to mask the matrix
    bounds : tuple of (float, float) or None, default=(None, None)
        Tuple of (lower, upper) bounds for the auxiliary variable v.
        If None, the bounds are inferred from the data.
        In practice, one can also pass the expected bounds of the matrix
        (e.g. (0, 1) for cosine similarity)

    Attributes
    ----------
    w_ : np.ndarray of shape (n_samples, rank)
        Learned factor matrix w
    components_ : np.ndarray of shape (n_samples, rank)
        Alias for w_ (sklearn compatibility)
    n_iter_ : int
        Number of SRF iterations performed
    history_ : dict
        Dictionary containing optimization metrics per iteration

    Examples
    --------
    >>> # Basic usage with complete data
    >>> from pysrf import SRF
    >>> model = SRF(rank=10, random_state=42)
    >>> w = model.fit_transform(similarity_matrix)
    >>> reconstruction = w @ w.T

    >>> # Usage with missing data (NaN values)
    >>> similarity_matrix[mask] = np.nan
    >>> model = SRF(rank=10, missing_values=np.nan)
    >>> w = model.fit_transform(similarity_matrix)

    References
    ----------
    .. [1] Shi et al. (2016). "Inexact Block Coordinate Descent Methods For
           Symmetric Nonnegative Matrix Factorization"
    """

    _parameter_constraints = {
        "rank": [Interval(Integral, 1, None, closed="left")],
        "rho": [Interval(Real, 0.0, None, closed="left")],
        "max_outer": [Interval(Integral, 1, None, closed="left")],
        "max_inner": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "init": [StrOptions({"random", "random_sqrt"})],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],  # sklearn's special validator
        "missing_values": [None, Real, np.nan],
        "bounds": [None, tuple],
    }
    _solver = staticmethod(_update_w_impl)

    def __init__(
        self,
        rank: int = 10,
        rho: float = 3.0,
        max_outer: int = 30,
        max_inner: int = 20,
        tol: float = 1e-4,
        verbose: int = 0,
        init: str = "random_sqrt",
        random_state: int | None = None,
        missing_values: float | None = np.nan,
        bounds: tuple[float, float] | None = (None, None),
    ) -> None:
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.missing_values = missing_values
        self.bounds = bounds

    def _compute_metrics(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_hat: np.ndarray,
        lam: np.ndarray,
        primal_residual: np.ndarray,
        v_old: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute ADMM optimization metrics for monitoring and convergence.

        Parameters
        ----------
        x : ndarray of shape (n, n)
            Original similarity matrix
        v : ndarray of shape (n, n)
            Auxiliary variable
        x_hat : ndarray of shape (n, n)
            Current estimate WW'
        lam : ndarray of shape (n, n)
            Lagrange multipliers
        primal_residual : ndarray of shape (n, n)
            V - WW'
        v_old : ndarray of shape (n, n) or None
            Previous V, used to compute the dual residual

        Returns
        -------
        metrics : dict[str, float]
            Keys: total_objective, data_fit, penalty, lagrangian,
            rec_error, evar, primal_residual, dual_residual
        """
        data_residual = x - v

        observed_mask = self._observation_mask
        data_fit = np.sum(data_residual[observed_mask] ** 2)

        penalty_term = np.sum(primal_residual**2)
        penalty = (self.rho / 2.0) * penalty_term

        lagrangian = np.sum(lam * primal_residual)
        total_obj = data_fit + penalty + lagrangian

        rec_residual_obs = (data_residual + primal_residual)[observed_mask]
        rec_ss = np.sum(rec_residual_obs**2)
        rec_error = np.sqrt(rec_ss)

        if self._total_var > 0:
            evar = 1.0 - rec_ss / self._total_var
        else:
            evar = 0.0

        primal_res = np.sqrt(penalty_term)
        dual_res = self.rho * np.linalg.norm(v - v_old) if v_old is not None else np.inf

        return {
            "total_objective": total_obj,
            "data_fit": data_fit,
            "penalty": penalty,
            "lagrangian": lagrangian,
            "rec_error": rec_error,
            "evar": evar,
            "primal_residual": primal_res,
            "dual_residual": dual_res,
        }

    def _check_convergence(
        self,
        primal_res: float,
        dual_res: float,
        v: np.ndarray,
        x_hat: np.ndarray,
        lam: np.ndarray,
    ) -> bool:
        """Check ADMM convergence using primal and dual residual norms.

        Parameters
        ----------
        primal_res : float
            Norm of the primal residual ||V - WW'||_F
        dual_res : float
            Norm of the dual residual rho * ||V - V_old||_F
        v : ndarray of shape (n, n)
            Current auxiliary variable
        x_hat : ndarray of shape (n, n)
            Current estimate WW'
        lam : ndarray of shape (n, n)
            Current Lagrange multipliers

        Returns
        -------
        converged : bool
            True if both primal and dual tolerances are satisfied
        """
        eps_abs = self.tol
        eps_rel = 1e-4
        n = v.size

        eps_pri = np.sqrt(n) * eps_abs + eps_rel * max(
            np.linalg.norm(v), np.linalg.norm(x_hat)
        )
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(lam)
        return primal_res <= eps_pri and dual_res <= eps_dual

    def _fit_complete_data(self, x: np.ndarray) -> SRF:
        """Fit model with complete data (no missing values).

        Uses _frobenius_residual to compute ||X - WW'||_F without forming WW',
        so memory stays at O(n^2).
        """
        w = _initialize_w(x, self.rank, self.init, self.random_state)
        history = defaultdict(list)

        n = x.shape[0]
        x_norm = np.linalg.norm(x, "fro")
        total_var = np.var(x)
        eps_rel = 1e-4

        pbar = trange(
            1,
            self.max_outer + 1,
            disable=not self.verbose,
            desc="SRF",
            mininterval=10.0 if not sys.stderr.isatty() else 0.1,
        )
        for i in pbar:
            w = self._solver(x, w, max_iter=self.max_inner, tol=self.tol)

            residual_norm, xhat_norm = _frobenius_residual(x, w)

            rec_error = residual_norm
            mse = residual_norm**2 / (n * n)
            evar = 1.0 - mse / total_var if total_var > 0 else 0.0

            history["rec_error"].append(rec_error)
            history["evar"].append(evar)
            history["primal_residual"].append(residual_norm)

            pbar.set_postfix(
                rec_error=f"{rec_error:.3f}",
                evar=f"{evar:.3f}",
            )

            eps_pri = n * self.tol + eps_rel * max(x_norm, xhat_norm)
            if residual_norm <= eps_pri:
                logger.info("Converged at iteration %d", i)
                break

        self._store_results(w, i, history)

        return self

    def _fit_missing_data(self, x: np.ndarray) -> SRF:
        """Fit model with missing data using ADMM."""
        bound_min, bound_max = self.bounds
        history = defaultdict(list)

        w = _initialize_w(x, self.rank, self.init, self.random_state)
        lam = np.zeros_like(x)

        v = x.copy()
        x_hat = w @ w.T

        v_old = np.empty_like(x)
        target = np.empty_like(x)
        primal_residual = np.empty_like(x)

        pbar = trange(
            1,
            self.max_outer + 1,
            disable=not self.verbose,
            desc="SRF",
            mininterval=10.0 if not sys.stderr.isatty() else 0.1,
        )
        for i in pbar:
            v_old[:] = v

            np.divide(lam, self.rho, out=target)
            np.add(v, target, out=target)
            w = self._solver(target, w, max_iter=self.max_inner, tol=self.tol)
            x_hat[:] = w @ w.T

            update_v_(
                self._observation_mask, x, x_hat, lam, self.rho, bound_min, bound_max, v
            )
            np.subtract(v, x_hat, out=primal_residual)
            lam += self.rho * primal_residual

            metrics = self._compute_metrics(x, v, x_hat, lam, primal_residual, v_old)
            for key, value in metrics.items():
                history[key].append(value)

            pbar.set_postfix(
                obj=f"{metrics['total_objective']:.3f}",
                rec_error=f"{metrics['rec_error']:.3f}",
                evar=f"{metrics['evar']:.3f}",
            )

            if i > 1 and self._check_convergence(
                metrics["primal_residual"],
                metrics["dual_residual"],
                v,
                x_hat,
                lam,
            ):
                logger.info("Converged at iteration %d", i)
                break

        self._store_results(w, i, history)

        return self

    def _store_results(self, w: np.ndarray, n_iter: int, history: dict) -> None:
        """Set fitted attributes after optimization."""
        self.w_ = w
        self.components_ = w
        self.n_iter_ = n_iter
        self.history_ = history

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> SRF:
        """
        Fit the symmetric NMF model to the data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values are allowed and should
            be marked according to the missing_values parameter.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        _validate_bounds(self.bounds)

        x = validate_data(
            self,
            x,
            reset=True,
            ensure_all_finite=(
                "allow-nan" if _is_nan_marker(self.missing_values) else True
            ),
            ensure_2d=True,
            dtype=np.float64,
            copy=True,
        )

        self._missing_mask = _get_missing_mask(x, self.missing_values)
        if np.all(self._missing_mask):
            raise ValueError(
                "No observed entries found in the data. All values are missing."
            )

        check_symmetric(self._missing_mask, raise_exception=True)
        self._observation_mask = ~self._missing_mask
        x[self._missing_mask] = 0.0
        x = check_symmetric(x, raise_exception=True, tol=1e-10)

        observed = self._observation_mask
        observed_count = int(np.sum(observed))
        if observed_count > 0:
            observed_mean = np.sum(x[observed]) / observed_count
            self._total_var = np.sum((x[observed] - observed_mean) ** 2)
        else:
            self._total_var = 0.0

        if np.all(self._observation_mask):
            return self._fit_complete_data(x)
        else:
            return self._fit_missing_data(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project data onto the learned factor space.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric matrix to transform

        Returns
        -------
        w : array-like of shape (n_samples, rank)
            Transformed data
        """
        check_is_fitted(self)
        return self.w_

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit the model and return the learned factors.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        w : array-like of shape (n_samples, rank)
            Learned factor matrix
        """
        return self.fit(x, y).transform(x)

    def reconstruct(self, w: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruct the similarity matrix from factors.

        Parameters
        ----------
        w : array-like of shape (n_samples, rank) or None
            Factor matrix to use for reconstruction.
            If None, uses the fitted factors.

        Returns
        -------
        s_hat : array-like of shape (n_samples, n_samples)
            Reconstructed similarity matrix
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_

        return w @ w.T

    def score(self, x: np.ndarray, y: np.ndarray | None = None) -> float:
        """
        Score the model using reconstruction error on observed entries only.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values are allowed and should
            be marked according to the missing_values parameter.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        mse : float
            Mean squared error of the reconstruction on observed entries.
        """
        check_is_fitted(self)

        x = validate_data(
            self,
            x,
            reset=False,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite="allow-nan" if self.missing_values is np.nan else True,
        )
        observation_mask = ~_get_missing_mask(x, self.missing_values)
        reconstruction = self.reconstruct()
        mse = np.mean((x[observation_mask] - reconstruction[observation_mask]) ** 2)
        return -mse
