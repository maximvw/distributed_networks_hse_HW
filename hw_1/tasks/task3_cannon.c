#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static inline int mod(int a,int b){ int r=a%b; return r<0? r+b: r; }

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc<2){ if(rank==0) fprintf(stderr,"Usage: %s <N>\n",argv[0]); MPI_Finalize(); return 1; }
    int N = atoi(argv[1]);

    int q = (int)round(sqrt(size));
    if (q*q != size) {
        if(rank==0) fprintf(stderr,"Number of processes must be a perfect square (q*q). Got %d.\n", size);
        MPI_Finalize(); return 1;
    }
    if (N % q != 0) {
        if(rank==0) fprintf(stderr,"Matrix size N must be divisible by sqrt(P)=%d.\n", q);
        MPI_Finalize(); return 1;
    }

    int block = N / q;
    int my_row = rank / q;
    int my_col = rank % q;

    int dims[2] = {q, q};
    int periods[2] = {1,1};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    int up, down, left, right;
    MPI_Cart_shift(grid, 0, 1, &up, &down);
    MPI_Cart_shift(grid, 1, 1, &left, &right);
    double *A = malloc(sizeof(double) * block * block);
    double *B = malloc(sizeof(double) * block * block);
    double *C = calloc(block * block, sizeof(double));
    int row0 = my_row * block;
    int col0 = my_col * block;
    for(int i=0;i<block;i++){
        for(int j=0;j<block;j++){
            int gi = row0 + i;
            int gj = col0 + j;
            A[i*block + j] = (double)(gi+1) + 0.1*(gj+1);
            B[i*block + j] = (double)(gi+1) - 0.2*(gj+1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for(int s=0;s<my_row;s++){
        MPI_Sendrecv_replace(A, block*block, MPI_DOUBLE, left, 0, right, 0, grid, MPI_STATUS_IGNORE);
    }
    for(int s=0;s<my_col;s++){
        MPI_Sendrecv_replace(B, block*block, MPI_DOUBLE, up, 0, down, 0, grid, MPI_STATUS_IGNORE);
    }

    for(int step=0; step<q; step++){
        for(int i=0;i<block;i++){
            for(int k=0;k<block;k++){
                double a = A[i*block + k];
                for(int j=0;j<block;j++){
                    C[i*block + j] += a * B[k*block + j];
                }
            }
        }
        MPI_Sendrecv_replace(A, block*block, MPI_DOUBLE, left, 0, right, 0, grid, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B, block*block, MPI_DOUBLE, up, 0, down, 0, grid, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if(rank==0){
        double time = t1 - t0;
        printf("N,procs,time\n");
        printf("%d,%d,%.6f\n", N, size, time);
    }

    free(A); free(B); free(C);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
