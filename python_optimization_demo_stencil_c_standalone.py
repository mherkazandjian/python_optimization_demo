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
#mat = numpy.repeat(numpy.repeat(mat, 4, axis=0), 4, axis=1)
m, n = mat.shape

### prepare the data (flatten it)
# %%
m, n = mat.shape
mat_flat = mat.flatten()
mat_edges_flat = numpy.zeros_like(mat_flat)

time_results = {}

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

# the number of flops is 5 per element
flops = 5
flops_total = m * n * flops

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
