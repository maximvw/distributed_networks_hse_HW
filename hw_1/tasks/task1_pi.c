#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc < 2) {
        if (rank==0) fprintf(stderr,"Usage: %s <num_points_per_proc> [seed]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long long local_n = atoll(argv[1]);
    unsigned int seed = (argc>=3) ? (unsigned int)atoi(argv[2]) : (unsigned int)(time(NULL) ^ (rank*7919));
    // use rand_r
    long long local_hits = 0;
    double t0 = MPI_Wtime();

    for (long long i=0;i<local_n;i++) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if (x*x + y*y <= 1.0) local_hits++;
    }

    long long total_hits = 0;
    long long total_points = 0;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_n, &total_points, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    if (rank==0) {
        double pi_est = 4.0 * (double)total_hits / (double)total_points;
        double time = t1 - t0;
        // printf("pi_est,points,procs,time\n");
        printf("% .12f,%lld,%d,%.6f\n", pi_est, total_points, size, time);
        // printf("Estimated pi = %.12f\n", pi_est);
        // printf("Total points = %lld, processes = %d, time = %.6f s\n", total_points, size, time);
    }

    MPI_Finalize();
    return 0;
}
