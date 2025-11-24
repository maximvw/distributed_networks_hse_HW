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
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "N must be divisible by number of processes for this test (N %% P == 0).\n");
        MPI_Finalize();
        return 1;
    }

    int cols_per_proc = N / size;

    double *A_local = malloc(sizeof(double) * N * cols_per_proc);
    double *x_local = malloc(sizeof(double) * cols_per_proc);
    double *y_local = calloc(N, sizeof(double));

    if (!A_local || !x_local || !y_local) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N * cols_per_proc; i++) A_local[i] = 1.0;
    for (int j = 0; j < cols_per_proc; j++) x_local[j] = 1.0;

    for (int local_col = 0; local_col < cols_per_proc; local_col++) {
        double xv = x_local[local_col];
        for (int i = 0; i < N; i++) {
            y_local[i] += A_local[i * cols_per_proc + local_col] * xv;
        }
    }

    double *y_global = NULL;
    if (rank == 0) y_global = malloc(sizeof(double) * N);
    MPI_Reduce(y_local, y_global, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int ok = 1;
        for (int i = 0; i < N; i++) {
            if (fabs(y_global[i] - (double)N) > 1e-9) {
                printf("test_task2_matvec_cols failed at i=%d: got %.12f expected %.12f\n",
                       i, y_global[i], (double)N);
                ok = 0;
                break;
            }
        }
        if (ok) printf("test_task2_matvec_cols passed (N=%d, procs=%d)\n", N, size);
        free(y_global);
    }

    free(A_local);
    free(x_local);
    free(y_local);
    MPI_Finalize();
    return 0;
}
