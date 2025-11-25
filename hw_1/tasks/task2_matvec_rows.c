#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2){
        if(rank == 0) fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    int N = atoi(argv[1]);

    // 1. Декомпозиция (Разбиение по строкам)
    int rows_per = N / size;
    int rem = N % size;
    // Определяем, какие строки достались этому процессу
    int my_first = rank * rows_per + (rank < rem ? rank : rem);
    int my_rows = rows_per + (rank < rem ? 1 : 0);

    // 2. Выделение памяти
    // A: храним только полосу строк (my_rows x N)
    double *A = malloc(sizeof(double) * my_rows * N);
    // x: нужен ЦЕЛИКОМ на каждом процессе
    double *x = malloc(sizeof(double) * N);
    // y_local: кусок результирующего вектора (только my_rows элементов)
    double *y_local = malloc(sizeof(double) * my_rows);

    // 3. Заполнение данных (Детерминированное)
    // Заполняем свою полосу матрицы формулой A[i][j] = i + j
    for(int i=0; i<my_rows; i++){
        int global_i = my_first + i;
        for(int j=0; j<N; j++){
            A[i*N + j] = (double)(global_i + j);
        }
    }
    
    // Вектор x заполняет только главный процесс
    if(rank == 0) {
        for(int j=0; j<N; j++){ 
            x[j] = 1.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 4. Коммуникация 1: Рассылка вектора x всем
    // При строчном разбиении каждому процессу нужен ВЕСЬ вектор x
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 5. Вычисления
    // Скалярное произведение строк матрицы на вектор x
    for(int i=0; i<my_rows; i++){
        double sum = 0.0;
        for(int j=0; j<N; j++) {
            sum += A[i*N + j] * x[j];
        }
        y_local[i] = sum;
    }

    // 6. Коммуникация 2: Сборка результатов
    double *y = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if(rank == 0){
        y = malloc(sizeof(double) * N);
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        
        // Восстанавливаем логику разбиения, чтобы знать, сколько от кого ждать
        int offset = 0;
        for(int p=0; p<size; p++){
            int r_rows = N/size + (p < (N%size) ? 1 : 0);
            recvcounts[p] = r_rows;
            displs[p] = offset;
            offset += r_rows;
        }
    }

    // Собираем куски y_local в один большой массив y на Rank 0
    // Gatherv нужен, так как куски могут быть разного размера
    MPI_Gatherv(y_local, my_rows, MPI_DOUBLE, 
                y, recvcounts, displs, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    double t1 = MPI_Wtime();

    // 7. Проверка результатов (только на Rank 0)
    if(rank == 0){
        int errors = 0;
        // Предвычисленная сумма арифметической прогрессии 0..N-1
        double arithmetic_sum_j = (double)N * (N - 1) / 2.0;

        for(int i=0; i<N; i++){
            // Ожидаемое значение для i-й строки: N*i + sum(j)
            double expected = (double)N * i + arithmetic_sum_j;
            
            if(fabs(y[i] - expected) > 1e-1) {
                errors++;
                // printf("Mismatch at index %d: expected %.2f, got %.2f\n", i, expected, y[i]);
            }
        }

        if(errors == 0){
            printf("OK,%d,%d,%.6f\n", N, size, t1 - t0);
        } else {
            printf("FAIL,%d,%d,%d\n", N, size, errors);
        }
        
        free(y); 
        free(recvcounts); 
        free(displs);
    }

    free(A); 
    free(x); 
    free(y_local);
    MPI_Finalize();
    return 0;
}