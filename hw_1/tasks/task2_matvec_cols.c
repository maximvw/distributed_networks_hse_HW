#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc<2){ if(rank==0) fprintf(stderr,"Usage: %s <N>\n",argv[0]); MPI_Finalize(); return 1; }
    int N = atoi(argv[1]);

    int cols_per = N / size;
    int rem = N % size;
    int my_first = rank * cols_per + (rank < rem ? rank : rem);
    int my_cols = cols_per + (rank < rem ? 1 : 0);

    double *A = malloc(sizeof(double) * N * my_cols);
    double *x_local = malloc(sizeof(double) * my_cols);
    double *y_local = malloc(sizeof(double) * N);
    double *y = malloc(sizeof(double) * N);

    for(int j=0;j<my_cols;j++){
        int gj = my_first + j;
        x_local[j] = 1.0;
        for(int i=0;i<N;i++){
            A[i*my_cols + j] = (double)(i+1) + 0.1*(gj+1);
        }
    }
    for(int i=0;i<N;i++) y_local[i] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for(int j=0;j<my_cols;j++){
        for(int i=0;i<N;i++){
            y_local[i] += A[i*my_cols + j] * x_local[j];
        }
    }

    MPI_Reduce(y_local, y, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if(rank==0){
        double time = t1 - t0;
        printf("N,procs,time\n");
        printf("%d,%d,%.6f\n", N, size, time);
    }

    free(A); free(x_local); free(y_local); free(y);
    MPI_Finalize();
    return 0;
}
