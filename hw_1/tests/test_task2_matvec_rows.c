#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 8;
    int rows_per_proc = N / size;

    double *A = malloc(rows_per_proc * N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    double *y_local = malloc(rows_per_proc * sizeof(double));

    for (int i = 0; i < N; i++) x[i] = 1.0;
    for (int i = 0; i < rows_per_proc * N; i++) A[i] = 1.0;

    for (int i = 0; i < rows_per_proc; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < N; j++)
            y_local[i] += A[i * N + j] * x[j];
    }

    double *y_global = NULL;
    if (rank == 0) y_global = malloc(N * sizeof(double));
    MPI_Gather(y_local, rows_per_proc, MPI_DOUBLE, y_global, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            if (fabs(y_global[i] - N) > 1e-9) {
                printf("test_task2_matvec_rows failed at i=%d: %f != %f\n", i, y_global[i], (double)N);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        printf("test_task2_matvec_rows passed (N=%d, procs=%d)\n", N, size);
        free(y_global);
    }

    free(A); free(x); free(y_local);
    MPI_Finalize();
    return 0;
}
