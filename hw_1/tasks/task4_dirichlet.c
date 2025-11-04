/* task4_dirichlet.c
   Solve Poisson (Laplace with source) on unit square with Dirichlet BC using Gauss-Seidel wavefront (block decomposition).
   Usage: mpirun -np <P> ./task4_dirichlet <N> <itmax> <px> <py>
   - N: number of grid points in each dimension (including boundaries). N >= 3.
   - itmax: iterations
   - px, py: process grid dimensions (px*py == P)
   Output CSV: N,procs,px,py,itmax,time,it_per_sec
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(i,j,nx) ((i)*(nx)+(j))

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc<5){
        if(rank==0) fprintf(stderr,"Usage: %s <N> <itmax> <px> <py>\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    int N = atoi(argv[1]); int itmax = atoi(argv[2]);
    int px = atoi(argv[3]); int py = atoi(argv[4]);
    if (px*py != size){ if(rank==0) fprintf(stderr,"px*py must equal number of processes\n"); MPI_Finalize(); return 1; }
    if (N < 3) { if(rank==0) fprintf(stderr,"N must be >=3\n"); MPI_Finalize(); return 1; }

    double bc = 100.0; // boundary constant c
    // domain: (0..N-1) indices are grid points including boundaries
    // split interior points among px x py blocks approximately
    int interior = N - 2;
    int irows = interior / px; int irem = interior % px;
    int icols = interior / py; int jrem = interior % py;

    int pr = rank / py; int pc = rank % py;
    int local_i0 = pr * irows + (pr < irem ? pr : irem);
    int local_ni = irows + (pr < irem ? 1 : 0);
    int local_j0 = pc * icols + (pc < jrem ? pc : jrem);
    int local_nj = icols + (pc < jrem ? 1 : 0);

    // allocate (local_ni + 2) x (local_nj + 2) including ghosts
    int nx = local_nj + 2;
    int ny = local_ni + 2;
    double *u = malloc(sizeof(double) * nx * ny);
    // init: set boundary values
    for(int i=0;i<ny;i++) for(int j=0;j<nx;j++) u[IDX(i,j,nx)] = 0.0;

    // set ghost boundaries corresponding to global boundaries
    // if process touches a global boundary, fill the corresponding ghost row/col with bc (Dirichlet)
    int global_i_start = local_i0 + 1; // first interior global index
    int global_j_start = local_j0 + 1;

    int top_global = (global_i_start == 1);
    int bottom_global = (global_i_start + local_ni - 1 == N-2);
    int left_global = (global_j_start == 1);
    int right_global = (global_j_start + local_nj - 1 == N-2);

    // set boundary ghost layers to bc for outer boundaries
    if (top_global) for(int j=0;j<nx;j++) u[IDX(0,j,nx)] = bc;
    if (bottom_global) for(int j=0;j<nx;j++) u[IDX(ny-1,j,nx)] = bc;
    if (left_global) for(int i=0;i<ny;i++) u[IDX(i,0,nx)] = bc;
    if (right_global) for(int i=0;i<ny;i++) u[IDX(i,nx-1,nx)] = bc;

    // neighbors in process grid
    int up = (pr==0) ? MPI_PROC_NULL : (rank - py);
    int down = (pr==px-1) ? MPI_PROC_NULL : (rank + py);
    int left = (pc==0) ? MPI_PROC_NULL : (rank - 1);
    int right = (pc==py-1) ? MPI_PROC_NULL : (rank + 1);

    // prepare MPI datatypes for row and column exchanges
    MPI_Datatype column_type;
    MPI_Type_vector(local_ni, 1, nx, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // buffers for sending/receiving ghost rows/cols
    double *send_top = malloc(sizeof(double)*local_nj);
    double *send_bottom = malloc(sizeof(double)*local_nj);
    double *recv_top = malloc(sizeof(double)*local_nj);
    double *recv_bottom = malloc(sizeof(double)*local_nj);
    double *send_left = malloc(sizeof(double)*local_ni);
    double *send_right = malloc(sizeof(double)*local_ni);
    double *recv_left = malloc(sizeof(double)*local_ni);
    double *recv_right = malloc(sizeof(double)*local_ni);

    // initialize interior u to 0
    for(int i=1;i<ny-1;i++) for(int j=1;j<nx-1;j++) u[IDX(i,j,nx)] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for(int it=0; it<itmax; it++){
        // Exchange ghost layers (simple synchronous exchange)
        // send top interior row (i=1) to up process's bottom ghost
        for(int j=0;j<local_nj;j++) send_top[j] = u[IDX(1,j+1,nx)];
        for(int j=0;j<local_nj;j++) send_bottom[j] = u[IDX(local_ni,j+1,nx)];
        for(int i=0;i<local_ni;i++) send_left[i] = u[IDX(i+1,1,nx)];
        for(int i=0;i<local_ni;i++) send_right[i] = u[IDX(i+1,local_nj,nx)];

        MPI_Request reqs[8];
        int rc=0;
        // top<->up (send_top to up, recv_top from up into ghost row 0)
        MPI_Isend(send_top, local_nj, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[rc++]);
        MPI_Irecv(recv_top, local_nj, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[rc++]);
        // bottom<->down
        MPI_Isend(send_bottom, local_nj, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[rc++]);
        MPI_Irecv(recv_bottom, local_nj, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[rc++]);
        // left<->left
        MPI_Isend(send_left, local_ni, MPI_DOUBLE, left, 2, MPI_COMM_WORLD, &reqs[rc++]);
        MPI_Irecv(recv_left, local_ni, MPI_DOUBLE, left, 3, MPI_COMM_WORLD, &reqs[rc++]);
        // right<->right
        MPI_Isend(send_right, local_ni, MPI_DOUBLE, right, 3, MPI_COMM_WORLD, &reqs[rc++]);
        MPI_Irecv(recv_right, local_ni, MPI_DOUBLE, right, 2, MPI_COMM_WORLD, &reqs[rc++]);

        MPI_Waitall(rc, reqs, MPI_STATUSES_IGNORE);

        // copy received data into ghost layers (if neighbor exists)
        if (up != MPI_PROC_NULL) {
            for(int j=0;j<local_nj;j++) u[IDX(0,j+1,nx)] = recv_top[j];
        }
        if (down != MPI_PROC_NULL) {
            for(int j=0;j<local_nj;j++) u[IDX(local_ni+1,j+1,nx)] = recv_bottom[j];
        }
        if (left != MPI_PROC_NULL) {
            for(int i=0;i<local_ni;i++) u[IDX(i+1,0,nx)] = recv_left[i];
        }
        if (right != MPI_PROC_NULL) {
            for(int i=0;i<local_ni;i++) u[IDX(i+1,local_nj+1,nx)] = recv_right[i];
        }

        // Gauss-Seidel style: update points in lexicographic order within block
        double maxdiff = 0.0;
        for(int i=1;i<=local_ni;i++){
            for(int j=1;j<=local_nj;j++){
                // index in global grid: gi = local_i0 + (i-1) +1 (1-based interior)
                // Laplacian discretization with grid spacing h = 1/(N-1)
                double old = u[IDX(i,j,nx)];
                double newv = 0.25 * ( u[IDX(i-1,j,nx)] + u[IDX(i+1,j,nx)] + u[IDX(i,j-1,nx)] + u[IDX(i,j+1,nx)] );
                u[IDX(i,j,nx)] = newv;
                double diff = fabs(newv - old);
                if (diff > maxdiff) maxdiff = diff;
            }
        }

        // compute global max diff to monitor convergence (optional)
        double global_maxdiff;
        MPI_Allreduce(&maxdiff, &global_maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        // Could break if global_maxdiff < tol; but we run fixed iterations for reproducibility
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if(rank==0){
        double time = t1 - t0;
        printf("N,procs,px,py,itmax,time\n");
        printf("%d,%d,%d,%d,%d,%.6f\n", N, size, px, py, itmax, time);
    }

    MPI_Type_free(&column_type);
    free(u);
    free(send_top); free(send_bottom); free(recv_top); free(recv_bottom);
    free(send_left); free(send_right); free(recv_left); free(recv_right);

    MPI_Finalize();
    return 0;
}
