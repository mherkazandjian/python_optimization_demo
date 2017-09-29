"""
<keywords>
test, python
</keywords>
<description>
</description>
<seealso>
</seealso>
"""
import random


def generate_zeros_matrix(m, n):
    return [[0 for _ in range(n)] for _ in range(m)]


def generate_random_matrix(m, n):
    return [[random.random() for _ in range(n)] for _ in range(m)]


def laplacian_filter(mat):
    m, n = len(mat), len(mat[0])

    retval = generate_zeros_matrix(m, n)

    for r in range(1, m - 1):
        for c in range(1, n - 1):
            retval[r][c] = 0.25 * (
                mat[r-1][c] + mat[r+1][c] + mat[r][c - 1] + mat[r][c + 1]
            )

    return retval


def laplacian_filter_column_sweep(mat):
    m, n = len(mat), len(mat[0])

    retval = generate_zeros_matrix(m, n)

    for c in range(1, n - 1):
        for r in range(1, m - 1):
            retval[r][c] = 0.25 * (
                mat[r-1][c] + mat[r+1][c] + mat[r][c - 1] + mat[r][c + 1]
            )

    return retval



mat = generate_random_matrix(3, 5)
# mat = generate_random_matrix(2000, 2000)

# %timeit laplacian_filter(mat)
# >>> 1 loop, best of 3: 1.66 s per loop
laplacian_filter(mat)

# %timeit laplacian_filter_column_sweep(mat)
# >>> 1 loop, best of 3: 1.86 s per loop
laplacian_filter_column_sweep(mat)


print('done')
