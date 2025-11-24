#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* matmul_seq(double *A, double *B, int N) {
    double *C = calloc(N*N, sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
    return C;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 4;
    int q = (int)sqrt(size);
    if (q*q != size) {
        if (rank == 0)
            printf("⚠️ Cannon test: number of processes must form a square (1,4,9,...)\n");
        MPI_Finalize();
        return 0;
    }

    int nloc = N / q;
    double *A = malloc(nloc*nloc*sizeof(double));
    double *B = malloc(nloc*nloc*sizeof(double));
    double *C = calloc(nloc*nloc,sizeof(double));

    for (int i = 0; i < nloc*nloc; i++) { A[i] = 1.0; B[i] = 1.0; }

    for (int i = 0; i < nloc; i++)
        for (int j = 0; j < nloc; j++)
            for (int k = 0; k < nloc; k++)
                C[i*nloc + j] += A[i*nloc + k] * B[k*nloc + j];

    double local_sum = 0.0;
    for (int i = 0; i < nloc*nloc; i++)
        local_sum += C[i];
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *Afull = malloc(N*N*sizeof(double));
        double *Bfull = malloc(N*N*sizeof(double));
        for (int i = 0; i < N*N; i++) { Afull[i] = 1.0; Bfull[i] = 1.0; }
        double *Cseq = matmul_seq(Afull, Bfull, N);

        double seq_sum = 0.0;
        for (int i = 0; i < N*N; i++) seq_sum += Cseq[i];

        double diff = fabs(seq_sum - global_sum);
        if (diff < 1e-9)
            printf("test_task3_cannon passed (N=%d, procs=%d)\n", N, size);
        else
            printf("test_task3_cannon failed: %.6f != %.6f (diff=%.6f)\n",
                   global_sum, seq_sum, diff);

        free(Afull); free(Bfull); free(Cseq);
    }

    free(A); free(B); free(C);
    MPI_Finalize();
    return 0;
}
