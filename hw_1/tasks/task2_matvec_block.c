#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc<2){ if(rank==0) fprintf(stderr,"Usage: %s <N>\n",argv[0]); MPI_Finalize(); return 1; }
    int N = atoi(argv[1]);

    // choose px x py (process grid) near square
    int px = (int)floor(sqrt(size));
    while (size % px != 0) px--;
    int py = size / px;

    int rx = rank / py; // row coord
    int ry = rank % py; // col coord

    int rows_per = N / px; int rrem = N % px;
    int cols_per = N / py; int crem = N % py;
    int my_row_first = rx * rows_per + (rx < rrem ? rx : rrem);
    int my_rows = rows_per + (rx < rrem ? 1 : 0);
    int my_col_first = ry * cols_per + (ry < crem ? ry : crem);
    int my_cols = cols_per + (ry < crem ? 1 : 0);

    // A_local: my_rows x my_cols, x_local: my_cols
    double *A = malloc(sizeof(double) * my_rows * my_cols);
    double *x_local = malloc(sizeof(double) * my_cols);
    double *y_local = calloc(my_rows, sizeof(double));
    double *y = NULL;
    if(rank==0) y = malloc(sizeof(double) * N);

    // fill local A and x_local deterministically (global indices)
    for(int i=0;i<my_rows;i++){
        int gi = my_row_first + i;
        for(int j=0;j<my_cols;j++){
            int gj = my_col_first + j;
            A[i*my_cols + j] = (double)(gi+1) + 0.1*(gj+1);
        }
    }
    for(int j=0;j<my_cols;j++) x_local[j] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // We need to form global y = A * x. Strategy:
    // 1) Each process computes partial y_local_segment = A_local * x_local
    // 2) Gather contributions for same global y rows across processes that own different column blocks.
    // To do that, each process sends its partial contributions for its row-block to the rank that is responsible for that row-block and then reduce-sum across column partners.
    // We'll use MPI_Allreduce on communicator created for processes with same rx (same row-blocks) to sum partial y parts.

    // compute partial y_local (size my_rows)
    for(int i=0;i<my_rows;i++){
        double s = 0.0;
        for(int j=0;j<my_cols;j++) s += A[i*my_cols + j] * x_local[j];
        y_local[i] = s;
    }

    // create communicator for same rx (row-block group) to sum across columns
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rx, ry, &row_comm);
    double *y_row = malloc(sizeof(double) * my_rows);
    MPI_Allreduce(y_local, y_row, my_rows, MPI_DOUBLE, MPI_SUM, row_comm);

    // Now gather final y_row pieces to rank 0
    // First rank in each row group with ry==0 will send its y_row (they are identical across row_comm) to rank 0 with proper displacement.
    int leader;
    MPI_Comm_rank(row_comm, &leader); // leader==0 inside row_comm corresponds to ry==0
    if(leader==0){
        int global_first_row = my_row_first; // belongs to rx block
        // send to root via MPI_Gatherv across leaders (we can use MPI_Gather using communicator of leaders but simpler: root receives directly from each leader)
    }

    // We'll gather using MPI_Gather of y_row from all leaders using a communicator of leaders.
    // Build a communicator of leaders: color = (ry==0 ? 0 : MPI_UNDEFINED)
    MPI_Comm leaders_comm;
    int is_leader = (ry==0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, is_leader, rx, &leaders_comm);

    if (ry==0) {
        // determine counts and displs on root rank 0 in MPI_COMM_WORLD
        int leaders_rank, leaders_size;
        MPI_Comm_rank(leaders_comm, &leaders_rank);
        MPI_Comm_size(leaders_comm, &leaders_size);
        // leaders are px in number, each has my_rows length (may vary)
        // gather lengths
        int *recvcounts = NULL;
        int *displs = NULL;
        if (rank==0){
            recvcounts = malloc(sizeof(int) * leaders_size);
            displs = malloc(sizeof(int) * leaders_size);
        }
        int my_rows_local = my_rows;
        MPI_Gather(&my_rows_local, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, leaders_comm);
        if (rank==0){
            displs[0]=0;
            for(int i=1;i<leaders_size;i++) displs[i]=displs[i-1]+recvcounts[i-1];
        }
        // gather y_row arrays
        MPI_Gatherv(y_row, my_rows, MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, 0, leaders_comm);

        if(rank==0) { free(recvcounts); free(displs); }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if(rank==0){
        double time = t1 - t0;
        printf("N,procs,time,px,py\n");
        printf("%d,%d,%.6f,%d,%d\n", N, size, time, px, py);
    }

    free(A); free(x_local); free(y_local); free(y_row);
    if(y) free(y);
    if(ry==0) MPI_Comm_free(&leaders_comm);
    MPI_Comm_free(&row_comm);

    MPI_Finalize();
    return 0;
}
