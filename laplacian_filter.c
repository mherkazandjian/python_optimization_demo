
#include <stdio.h>
#include <stdlib.h>

void laplacian_filter(double *mat, const int m, const int n, double *retval)
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
