# %% [markdown]
# WARNING:
# some packages will probably be upgraded. Some packages.
# Create a virtual env if you want to keep your packages intact.

# %%
!pip3 install --user pillow==11.1.0 numba==0.61.0
!pip3 install --user numpy==2.1.3 scipy==1.15.2
!pip3 install --user imageio==2.37.0

# %%
## Load an image (1000 x 1000 pixels) and display it
# %%
import numpy
from imageio import imread

mat = imread('sample.jpg').astype('f8')
# resize to 4000x4000
mat = numpy.repeat(numpy.repeat(mat, 4, axis=0), 4, axis=1)
m, n = mat.shape

# %%
%matplotlib inline
from matplotlib import pyplot
pyplot.imshow(mat, cmap='gray')
pyplot.colorbar()
pyplot.title('Grayscale Image')
pyplot.show()
# %%
# Pure Python Implementation

### utility function that generates a matrix as a list of lists
# %%
def generate_zeros_matrix(m, n):
    return [[0 for _ in range(n)] for _ in range(m)]
# %%
### function that apply the filter by iterating over columns
# %%
def laplacian_filter_pure_python(mat, retval):
    for c in range(1, n - 1):
        for r in range(1, m - 1):
            retval[r][c] = -4.0*mat[r][c] +     \
                                mat[r-1][c] +   \
                                mat[r+1][c] +   \
                                mat[r][c - 1] + \
                                mat[r][c + 1]

### function that apply the filter by iterating over rows
def laplacian_filter_pure_python_row_sweep(mat, retval):
    for r in range(1, m - 1):
        for c in range(1, n - 1):
            retval[r][c] = -4.0*mat[r][c] +      \
                                mat[r-1][c] +    \
                                mat[r+1][c] +    \
                                mat[r][c - 1] +  \
                                mat[r][c + 1]

# %%
# convert the numpy image to a list of lists
mat_list_of_lists = [[pixel for pixel in row] for row in mat]

# create the matrix that will hold the "filtered" image (with the edges)
mat_edges = generate_zeros_matrix(m, n)

# %%
# apply the filter and display the convolved image
laplacian_filter_pure_python(mat_list_of_lists, mat_edges)

# convert the list of lists back to a numpy array
mat_edges = numpy.array(mat_edges)

# Display the convolved image with a narrowed colorbar
pyplot.imshow(mat_edges > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_pure_python = %timeit -o laplacian_filter_pure_python(mat_list_of_lists, mat_edges)
time_pure_python = timeit_pure_python.best

# save the value into a dictionary
time_results = {'Pure Python': time_pure_python}
# %%
timeit_pure_python_row_sweep = %timeit -o laplacian_filter_pure_python_row_sweep(mat_list_of_lists, mat_edges)
time_row_sweep = timeit_pure_python_row_sweep.best

# save the value into a dictionary
time_results['Pure Python Row Sweep'] = time_row_sweep

# plot the results
pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.show()
# %%

# Numpy
# %%

## Iterate over the numpy array with python loops
# %%
def laplacian_filter_numpy_python_loops(mat, retval):
    m, n = mat.shape
    for r in range(1, m - 1):
        for c in range(1, n - 1):
            retval[r][c] = -4.0*mat[r,c] +     \
                                mat[r-1,c] +   \
                                mat[r+1,c] +   \
                                mat[r,c - 1] + \
                                mat[r,c + 1]
    return retval
# %%
mat_edges = numpy.zeros_like(mat)
laplacian_filter_numpy_python_loops(mat, mat_edges)
pyplot.imshow(mat_edges > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_numpy_python_loop = %timeit -o laplacian_filter_numpy_python_loops(mat, mat_edges)
time_numpy_python_loop = timeit_numpy_python_loop.best

# save the value into a dictionary
time_results['Numpy Python Loops'] = time_numpy_python_loop

# plot the results
pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=45)
pyplot.tight_layout()
pyplot.show()
# %%

## Numpy vectorized operations only (no explicit python loops)
# %%
from functools import reduce
def laplacian_filter_numpy_roll(mat):
    mat_ext = numpy.pad(mat, 1, 'constant')

    rolled = [
        numpy.roll(mat_ext, 1, 0),
        numpy.roll(mat_ext, -1, 0),
        numpy.roll(mat_ext, 1, 1),
        numpy.roll(mat_ext, -1, 1)
    ]
    retval = reduce(numpy.add, rolled)[1:-1, 1:-1] - 4.0*mat
    return retval
# %%
mat_edges = laplacian_filter_numpy_roll(mat)
pyplot.imshow(mat_edges > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_numpy_roll = %timeit -o laplacian_filter_numpy_roll(mat)
time_numpy_roll = timeit_numpy_roll.best

# save the value into a dictionary
time_results['Numpy Roll'] = time_numpy_roll

# plot the results in log scale
pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()
# %%

## Numpy operations only (lower memory footprint)
# %%
def laplacian_filter_numpy_map_reduce(mat):

    mat_ext = numpy.pad(mat, 1, 'constant')

    retval = reduce(
        numpy.add,
        map(lambda args: numpy.roll(
            mat_ext, shift=args[0],
            axis=args[1]),
            [[-1, 0], [1, 0], [-1, 1], [1, 1]])
    )

    return retval[1:-1, 1:-1] - 4.0*mat
# %%
mat_edges = laplacian_filter_numpy_map_reduce(mat)
pyplot.imshow(mat_edges > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_numpy_map_reduce =  %timeit -o laplacian_filter_numpy_map_reduce(mat)
time_numpy_map_reduce = timeit_numpy_map_reduce.best

# save the value into a dictionary
time_results['Numpy Map-Reduce'] = time_numpy_map_reduce

# plot the results in log scale
pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()
# %%
## Pure python code + numba

#https://github.com/numba/numba
# %%
from numba import jit

# %%

### only difference between "laplacian_filter_numpy_python_loops" is "@jit"
# %%
@jit
def laplacian_filter_numba(mat, retval):
    m, n = mat.shape
    for c in range(1, n - 1):
        for r in range(1, m - 1):
            retval[r, c] = -4.0*mat[r, c] +     \
                                mat[r-1, c] +   \
                                mat[r+1, c] +   \
                                mat[r, c - 1] + \
                                mat[r, c + 1]
# %%

### iterate over the rows instead of columns
# %%
@jit
def laplacian_filter_numba_row(mat, retval):
    m, n = mat.shape
    for r in range(1, m - 1):
        for c in range(1, n - 1):
            retval[r, c] = -4.0*mat[r, c] +     \
                                mat[r-1, c] +   \
                                mat[r+1, c] +   \
                                mat[r, c - 1] + \
                                mat[r, c + 1]
# %%
laplacian_filter_numba(mat, mat_edges)
pyplot.imshow(mat_edges > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_numba = %timeit -o laplacian_filter_numba(mat, mat_edges)
time_numba = timeit_numba.best

# save the value into a dictionary
time_results['Numba'] = time_numba

# %%
timeit_numpy_python_loop = %timeit -o laplacian_filter_numba_row(mat, mat_edges)
time_numba_python_loop = timeit_numpy_python_loop.best

time_results['Numba Python Loop'] = time_numba_python_loop

pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()
# %%
# Cython
#https://github.com/cython/cython
# %%
!cat cython_bindings/*.pyx
# %%
!cat cython_bindings/setup.py
# %%
!rm -fvr cython_bindings/*.so
!cd cython_bindings && python3 setup.py build_ext --inplace
# %%
from cython_bindings.cython_binding import laplacian_filter_py
# %%
### prepare the data (flatten it)
# %%
m, n = mat.shape
mat_flat = mat.flatten()
mat_edges_flat = numpy.zeros_like(mat_flat)
# %%
laplacian_filter_py(mat_flat, m, n, mat_edges_flat)
pyplot.imshow(mat_edges.reshape(m, n) > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_cython = %timeit -o laplacian_filter_py(mat_flat, m, n, mat_edges_flat)
time_cython = timeit_cython.best

# save the value into a dictionary
time_results['Cython'] = time_cython

pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()
# %%
# Ctypes
#https://docs.python.org/3/library/ctypes.html

# %%
## Using gcc

# %%
%%file laplacian_filter_gcc.c

#include <stdio.h>
#include <stdlib.h>

void laplacian_filter_gcc(double *mat, const int m, const int n, double *retval)
{
    for( int r = 1; r < m - 1; r++)
    {
        const int offset = r*n;

        for( int c = 1; c < n - 1; c++)
        {
            const int i = offset + c;
            const int i_top = offset + n + c;
            const int i_bottom = offset - n + c;
            const int i_left = offset + (c-1);
            const int i_right = offset + (c+1);

            retval[i] = -4.0*mat[i] + mat[i_bottom] + mat[i_top] + mat[i_left] + mat[i_right];
        }
    }
}
# %%
!gcc laplacian_filter_gcc.c -std=c99 -shared -fPIC -O3 -o laplacian_filter_gcc.so
# %%
!ls -l laplacian_filter_gcc.so
# %%
import os
import ctypes

# load the function from the shared library
clib = ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), 'laplacian_filter_gcc.so'))
clib.laplacian_filter_gcc.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]


def laplacian_filter_ctypes_gcc(mat, m, n, retval):
    mat_ptr = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    retval_ptr = retval.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    clib.laplacian_filter_gcc(mat_ptr, m, n, retval_ptr)
# %%
laplacian_filter_ctypes_gcc(mat_flat, m, n, mat_edges_flat)
pyplot.imshow(mat_edges_flat.reshape(m, n) > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_ctypes = %timeit -o laplacian_filter_ctypes_gcc(mat_flat, m, n, mat_edges_flat)
time_ctypes = timeit_ctypes.best

# save the value into a dictionary
time_results['Ctypes GCC'] = time_ctypes

pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()

# %%
%%file laplacian_filter_gcc_opt.c

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void laplacian_filter_gcc_opt(double *mat, const int m, const int n, double *retval)
{
    // Load constant for multiplication by -4
    const __m512d neg_four = _mm512_set1_pd(-4.0);

    for (int r = 1; r < m - 1; r++)
    {
        const int offset = r * n;

        // Process 8 elements at a time with AVX-512
        int c = 1;
        for (; c < n - 8; c += 8)
        {
            // Calculate indices
            const int i = offset + c;
            const int i_top = i + n;
            const int i_bottom = i - n;

            // Load center elements and multiply by -4
            __m512d center = _mm512_loadu_pd(&mat[i]);
            __m512d result = _mm512_mul_pd(center, neg_four);

            // Load and add top elements
            __m512d top = _mm512_loadu_pd(&mat[i_top]);
            result = _mm512_add_pd(result, top);

            // Load and add bottom elements
            __m512d bottom = _mm512_loadu_pd(&mat[i_bottom]);
            result = _mm512_add_pd(result, bottom);

            // Load and add left elements (shifted by 1)
            __m512d left = _mm512_loadu_pd(&mat[i - 1]);
            result = _mm512_add_pd(result, left);

            // Load and add right elements (shifted by 1)
            __m512d right = _mm512_loadu_pd(&mat[i + 1]);
            result = _mm512_add_pd(result, right);

            // Store the result
            _mm512_storeu_pd(&retval[i], result);
        }

        // Handle remaining elements with scalar code
        for (; c < n - 1; c++)
        {
            const int i = offset + c;
            const int i_top = i + n;
            const int i_bottom = i - n;
            const int i_left = i - 1;
            const int i_right = i + 1;

            retval[i] = -4.0 * mat[i] + mat[i_bottom] + mat[i_top] + mat[i_left] + mat[i_right];
        }
    }
}
# %%
## Using gcc
# %%
!gcc laplacian_filter_gcc_opt.c -std=c99 -shared -fPIC -O3 -o laplacian_filter_gcc_opt.so -mavx512f -march=znver4
# %%
!ls -l laplacian_filter_gcc_opt.so
# %%
import os
import ctypes

# load the function from the shared library
clib = ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), 'laplacian_filter_gcc_opt.so'))
clib.laplacian_filter_gcc_opt.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]

def laplacian_filter_ctypes_gcc_opt(mat, m, n, retval):
    mat_ptr = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    retval_ptr = retval.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    clib.laplacian_filter_gcc_opt(mat_ptr, m, n, retval_ptr)
# %%
laplacian_filter_ctypes_gcc_opt(mat_flat, m, n, mat_edges_flat)
pyplot.imshow(mat_edges_flat.reshape(m, n) > 0, cmap='gray')
pyplot.colorbar()
pyplot.title('Laplacian Filter - Edges Detected')
pyplot.tight_layout()
pyplot.show()
# %%
timeit_ctypes_opt = %timeit -o laplacian_filter_ctypes_gcc_opt(mat_flat, m, n, mat_edges_flat)
time_ctypes_opt = timeit_ctypes_opt.best

# save the value into a dictionary
time_results['Ctypes GCC opt'] = time_ctypes_opt

pyplot.bar(time_results.keys(), time_results.values())
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()

# %%
## Using assembly

## .. todo:: Add assembly implementation


# %%
# find the minimum time and plot all bars with the minimum time in red
min_time = min(time_results.values())
min_time_key = [key for key in time_results if time_results[key] == min_time][0]

# Plot all bars
colors = ['red' if key == min_time_key else 'blue' for key in time_results.keys()]
bars = pyplot.bar(time_results.keys(), time_results.values(), color=colors)

# Add text labels showing speedup relative to minimum time
for key, value in time_results.items():
    inv_speedup = value / min_time  # This is the relative slowdown factor
    pyplot.text(key, value, f'{int(inv_speedup)}x' if inv_speedup > 10 else f'{inv_speedup:.2f}x', ha='center')

pyplot.grid(axis='y')
pyplot.ylabel('Time (s)')
pyplot.title('Execution Time')
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.yscale('log')  # set y-axis to log scale
pyplot.show()

# %%
