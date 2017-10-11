import cython
import numpy
cimport numpy as np


cdef extern from "../laplacian_filter.h" nogil:

    void laplacian_filter(double * mat, const int m, const int n, double * retval);


@cython.boundscheck(False)
@cython.wraparound(False)
def laplacian_filter_py(
    np.ndarray[np.float64_t, ndim=1, mode="c"] mat,
    const int m, const int n,
    np.ndarray[np.float64_t, ndim=1, mode="c"] retval
    ):

    laplacian_filter(&mat[0], m, n, &retval[0])
