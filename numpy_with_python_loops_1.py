"""
<keywords>
test, python
</keywords>
<description>
</description>
<seealso>
</seealso>
"""
from functools import reduce
import numpy


def laplacian_filter(mat):

    # m, n = mat.shape
    # retval_check = numpy.zeros_like(mat)
    # for r in range(1, m - 1):
    #     for c in range(1, n - 1):
    #         retval_check[r][c] = 0.25 * (
    #             mat[r-1][c] + mat[r+1][c] + mat[r][c - 1] + mat[r][c + 1]
    #         )
    # retval = retval_check

    mat_ext = numpy.pad(mat, 1, 'constant')
    rolled = [numpy.roll(mat_ext, 1, 0), numpy.roll(mat_ext, -1, 0), numpy.roll(mat_ext, 1, 1), numpy.roll(mat_ext, -1, 1)]

    retval = 0.25 * reduce(
        numpy.add,
        # map(lambda args: numpy.roll(mc, shift=args[0], axis=args[1]), [[-1, 0], [1, 0], [-1, 1], [1, 1]])
        rolled
    )[1:-1, 1:-1]

    # numpy.testing.assert_allclose(
    #     retval_check[1:-1, 1:-1],
    #     retval[1:-1, 1:-1],
    #     rtol=1e-15
    # )

    return retval


mat = numpy.random.rand(1000, 1000)
# mat = numpy.random.rand(3, 5)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        mat[i, j] = numpy.random.randint(0, 100)

# %timeit laplacian_filter(mat)
# >>> 1 loop, best of 3: 1.66 s per loop
laplacian_filter(mat)

# %timeit laplacian_filter_column_sweep(mat)
# >>> 1 loop, best of 3: 1.86 s per loop


print('done')
