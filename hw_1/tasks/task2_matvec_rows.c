#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc<2){
        if(rank==0) fprintf(stderr,"Usage: %s <N>\n",argv[0]);
        MPI_Finalize(); return 1;
    }
    int N = atoi(argv[1]);

    int rows_per = N / size;
    int rem = N % size;
    int my_first = rank * rows_per + (rank < rem ? rank : rem);
    int my_rows = rows_per + (rank < rem ? 1 : 0);

    // allocate local A block (my_rows x N) and x and local y
    double *A = malloc(sizeof(double) * my_rows * N);
    double *x = malloc(sizeof(double) * N);
    double *y_local = malloc(sizeof(double) * my_rows);

    // init matrix and vector deterministically
    for(int i=0;i<my_rows;i++){
        int gi = my_first + i;
        for(int j=0;j<N;j++){
            A[i*N + j] = (double)(gi+1) + 0.1*(j+1);
        }
    }
    for(int j=0;j<N;j++) x[j] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Because x is needed by all, broadcast x from rank 0 (or allfill)
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // compute local y = A_local * x
    for(int i=0;i<my_rows;i++){
        double sum = 0.0;
        for(int j=0;j<N;j++) sum += A[i*N + j] * x[j];
        y_local[i] = sum;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Optionally gather results to rank 0 (not required, but for verification)
    if(rank==0){
        double *y = malloc(sizeof(double) * N);
        int *recvcounts = malloc(sizeof(int) * size);
        int *displs = malloc(sizeof(int) * size);
        for(int p=0;p<size;p++){
            int rp = N/size + (p < (N%size) ? 1 : 0);
            recvcounts[p] = rp;
        }
        displs[0] = 0;
        for(int p=1;p<size;p++) displs[p] = displs[p-1] + recvcounts[p-1];
        MPI_Gatherv(y_local, my_rows, MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(y); free(recvcounts); free(displs);
    } else {
        MPI_Gatherv(y_local, my_rows, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if(rank==0){
        double time = t1 - t0;
        printf("N,procs,time\n");
        printf("%d,%d,%.6f\n", N, size, time);
    }

    free(A); free(x); free(y_local);
    MPI_Finalize();
    return 0;
}
