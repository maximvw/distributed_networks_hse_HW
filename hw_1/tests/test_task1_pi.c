#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long total_points = (argc > 1) ? atoll(argv[1]) : 1000000;
    // Равномерно распределяем точки
    long long local_points = total_points / size + (rank < (total_points % size) ? 1 : 0);

    unsigned int seed = (unsigned int)(time(NULL) + rank * 12345);
    srand(seed);

    long long local_hits = 0;
    for (long long i = 0; i < local_points; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            local_hits++;
    }

    long long total_hits = 0;
    long long total_samples = 0;

    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_points, &total_samples, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi_est = 4.0 * (double)total_hits / (double)total_samples;
        double err = fabs(pi_est - M_PI);
        if (err < 0.01)
            printf("test_task1_pi passed: pi=%.6f (error=%.6f)\n", pi_est, err);
        else
            printf("test_task1_pi failed: pi=%.6f (error=%.6f)\n", pi_est, err);
    }

    MPI_Finalize();
    return 0;
}
