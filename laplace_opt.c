#include <stdlib.h>
#include <omp.h>

void laplace_opt(double *mat, const int m, const int n, double *retval) {
    // ...existing code for any required setup...
    for (int r = 1; r < m - 1; r++) {
        int offset = r * n;
        for (int c = 1; c < n - 1; c++) {
            int i = offset + c;
            // Using indices: top, bottom, left, right, with no border handling
            retval[i] = -4.0 * mat[i] + mat[i - n] + mat[i + n] + mat[i - 1] + mat[i + 1];
        }
    }
    // ...existing code for border handling if needed...
}
