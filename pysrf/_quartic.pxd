# cython: language_level=3
from libc.math cimport sqrt, cbrt


cdef inline double _quartic_root(double a, double b, double c, double d) nogil:
    """Solve cubic resolvent of the quartic subproblem (Shi et al. 2016, Eq. 11)."""
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
