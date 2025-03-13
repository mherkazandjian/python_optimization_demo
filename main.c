#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void laplace_opt(double *mat, const int m, const int n, double *retval);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mat.bin> <m> <n>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    size_t size = m * n;

    double *mat = malloc(size * sizeof(double));
    double *retval = malloc(size * sizeof(double));
    if (!mat || !retval) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    FILE *fp = fopen(input_file, "rb");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    if (fread(mat, sizeof(double), size, fp) != size) {
        fprintf(stderr, "Error reading input file\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    // Initialize output array if necessary
    for (size_t i = 0; i < size; i++) {
        retval[i] = 0.0;
    }

    int iterations = 1000;
    double start = omp_get_wtime();
    for (int iter = 0; iter < iterations; iter++) {
        laplace_opt(mat, m, n, retval);
    }
    double end = omp_get_wtime();
    double elapsed = (end - start);
    double flops_total = (double)m * n * 5 * iterations;
    double gflops = flops_total / elapsed / 1e9;
    printf("Total execution time over %d runs: %f s\n", iterations, elapsed);
    printf("GFLOPs: %f\n", gflops);

    fp = fopen("output.bin", "wb");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fwrite(retval, sizeof(double), size, fp);
    fclose(fp);

    free(mat);
    free(retval);
    return 0;
}
