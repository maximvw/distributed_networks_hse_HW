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
    int ITMAX = (argc > 2) ? atoi(argv[2]) : 500;

    int px = 1, py = size;
    for (int i = (int)sqrt(size); i >= 1; i--) {
        if (size % i == 0) {
            px = i;
            py = size / i;
            break;
        }
    }

    int nx = N / px;
    int ny = N / py;

    if (rank == 0)
        printf("# Process grid: %d x %d = %d processes\n", px, py, size);

    double *u = calloc((nx + 2) * (ny + 2), sizeof(double));
    double *u_new = calloc((nx + 2) * (ny + 2), sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int it = 0; it < ITMAX; it++) {
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                u_new[i * (ny + 2) + j] = 0.25 * (
                    u[(i - 1) * (ny + 2) + j] +
                    u[(i + 1) * (ny + 2) + j] +
                    u[i * (ny + 2) + (j - 1)] +
                    u[i * (ny + 2) + (j + 1)]
                );
            }
        }
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double total_time;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N,procs,px,py,itmax,time\n");
        printf("%d,%d,%d,%d,%d,%.6f\n", N, size, px, py, ITMAX, total_time);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}
