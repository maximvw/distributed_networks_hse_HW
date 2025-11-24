#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* matvec_seq(double *A, double *x, int N) {
    double *y = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
    }
    return y;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 8;

    int px = 1, py = size;
    for (int i = (int)sqrt(size); i >= 1; i--) {
        if (size % i == 0) { px = i; py = size / i; break; }
    }

    int rows = N / px;
    int cols = N / py;
    int block_size = rows * cols;

    double *A_local = malloc(block_size * sizeof(double));
    double *x_local = malloc(cols * sizeof(double));
    double *y_local = malloc(rows * sizeof(double));

    for (int i = 0; i < block_size; i++) A_local[i] = 1.0;
    for (int i = 0; i < cols; i++) x_local[i] = 1.0;

    for (int i = 0; i < rows; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < cols; j++)
            y_local[i] += A_local[i * cols + j] * x_local[j];
    }

    double *y_partial = NULL;
    if (rank == 0)
        y_partial = malloc(rows * size * sizeof(double));

    MPI_Gather(y_local, rows, MPI_DOUBLE, y_partial, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *y_final = calloc(N, sizeof(double));
        for (int i = 0; i < px; i++) {
            for (int j = 0; j < rows; j++) {
                double sum = 0.0;
                for (int k = 0; k < py; k++)
                    sum += y_partial[(i * py + k) * rows + j];
                y_final[i * rows + j] = sum;
            }
        }

        double *A_full = malloc(N * N * sizeof(double));
        double *x_full = malloc(N * sizeof(double));
        for (int i = 0; i < N*N; i++) A_full[i] = 1.0;
        for (int i = 0; i < N; i++) x_full[i] = 1.0;
        double *y_seq = matvec_seq(A_full, x_full, N);

        int ok = 1;
        for (int i = 0; i < N; i++) {
            if (fabs(y_final[i] - y_seq[i]) > 1e-9) {
                printf("test_task2_matvec_block failed at i=%d: %.9f != %.9f\n",
                       i, y_final[i], y_seq[i]);
                ok = 0;
                break;
            }
        }
        if (ok)
            printf("test_task2_matvec_block passed (N=%d, procs=%d)\n", N, size);

        free(A_full);
        free(x_full);
        free(y_seq);
        free(y_partial);
        free(y_final);
    }

    free(A_local);
    free(x_local);
    free(y_local);

    MPI_Finalize();
    return 0;
}
