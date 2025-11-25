#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <TOTAL_points> [seed]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Считываем ОБЩЕЕ количество точек
    long long total_n = atoll(argv[1]);

    // Рассчитываем нагрузку для текущего процесса
    long long local_n = total_n / size;
    long long remainder = total_n % size;

    // Распределяем остаток равномерно по первым процессам
    if (rank < remainder) {
        local_n++;
    }

    // Генерация seed (уникальный для каждого процесса)
    unsigned int seed = (argc >= 3) ? (unsigned int)atoi(argv[2]) + rank : (unsigned int)(time(NULL) ^ (rank * 7919));
    
    long long local_hits = 0;
    
    // Барьер для чистоты замера времени (все ждут, пока все подготовятся)
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (long long i = 0; i < local_n; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if (x*x + y*y <= 1.0) local_hits++;
    }

    long long total_hits = 0;
    
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        double pi_est = 4.0 * (double)total_hits / (double)total_n;
        double time = t1 - t0;
        printf("%.12f,%lld,%d,%.6f\n", pi_est, total_n, size, time);
    }

    MPI_Finalize();
    return 0;
}