#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 100;
    int ITMAX = (argc > 2) ? atoi(argv[2]) : 200;

    int px=1, py=size;
    for(int i=(int)sqrt(size); i>=1; i--) if(size % i == 0){ px=i; py=size/i; break; }

    int nx=N/px, ny=N/py;
    double *u=calloc((nx+2)*(ny+2),sizeof(double));
    double *u_new=calloc((nx+2)*(ny+2),sizeof(double));

    for(int it=0; it<ITMAX; it++){
        for(int i=1;i<=nx;i++){
            for(int j=1;j<=ny;j++){
                u_new[i*(ny+2)+j]=0.25*(
                    u[(i-1)*(ny+2)+j]+u[(i+1)*(ny+2)+j]+
                    u[i*(ny+2)+(j-1)]+u[i*(ny+2)+(j+1)]);
            }
        }
        double *tmp=u; u=u_new; u_new=tmp;
    }

    double local_sum=0.0;
    for(int i=1;i<=nx;i++)
        for(int j=1;j<=ny;j++)
            local_sum+=u[i*(ny+2)+j];
    double global_sum;
    MPI_Reduce(&local_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    if(rank==0){
        double avg = global_sum / (N*N);
        if(fabs(avg) < 1e-6)
            printf("test_task4_dirichlet passed (N=%d, procs=%d)\n",N,size);
        else
            printf("test_task4_dirichlet failed: avg=%f\n",avg);
    }

    free(u);free(u_new);
    MPI_Finalize();
    return 0;
}
