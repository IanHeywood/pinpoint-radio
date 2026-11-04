"""Optional numba acceleration.

If numba is installed, kernels decorated with `maybe_njit` are compiled (and
parallelised over rows with `prange`). If not, the very same Python source runs
unchanged (slowly) - which is what lets the test-suite check the kernel logic on
machines without numba.
"""

try:  # pragma: no cover - depends on environment
    import numba as _numba

    HAVE_NUMBA = True
    prange = _numba.prange

    def maybe_njit(parallel=False):
        def deco(fn):
            return _numba.njit(parallel=parallel, fastmath=False, cache=True, nogil=True)(fn)

        return deco

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    prange = range

    def maybe_njit(parallel=False):
        def deco(fn):
            return fn

        return deco
