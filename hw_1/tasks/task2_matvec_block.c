#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1000;

    int px = 1, py = size;
    for (int i = (int)sqrt(size); i >= 1; i--) {
        if (size % i == 0) {
            px = i;
            py = size / i;
            break;
        }
    }

    int rows_per_proc = N / px;
    int cols_per_proc = N / py;

    if (rank == 0)
        printf("# Process grid: %d x %d = %d processes\n", px, py, size);

    double *A = malloc(rows_per_proc * cols_per_proc * sizeof(double));
    double *x = malloc(cols_per_proc * sizeof(double));
    double *y_local = malloc(rows_per_proc * sizeof(double));

    for (int i = 0; i < rows_per_proc * cols_per_proc; i++)
        A[i] = 1.0;
    for (int i = 0; i < cols_per_proc; i++)
        x[i] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < cols_per_proc; j++) {
            y_local[i] += A[i * cols_per_proc + j] * x[j];
        }
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double total_time;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N,procs,time,px,py\n");
        printf("%d,%d,%.6f,%d,%d\n", N, size, total_time, px, py);
    }

    free(A);
    free(x);
    free(y_local);
    MPI_Finalize();
    return 0;
}
