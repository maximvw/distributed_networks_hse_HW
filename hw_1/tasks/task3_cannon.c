#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    
    int rank_world, size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 2){ 
        if(rank_world == 0) fprintf(stderr,"Usage: %s <N>\n", argv[0]); 
        MPI_Finalize(); 
        return 1; 
    }
    
    int N = atoi(argv[1]);
    int q = (int)round(sqrt(size));

    // Проверки входных данных
    if (q * q != size) {
        if(rank_world == 0) fprintf(stderr,"Error: Number of processes must be a perfect square (e.g., 4, 9, 16). Got %d.\n", size);
        MPI_Finalize(); return 1;
    }
    if (N % q != 0) {
        if(rank_world == 0) fprintf(stderr,"Error: Matrix size N must be divisible by sqrt(P)=%d.\n", q);
        MPI_Finalize(); return 1;
    }

    // 1. Создаем топологию решетки
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; // Торус (цикличность)
    int reorder = 1;
    MPI_Comm grid;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid);

    // 2. Получаем координаты в решетке
    int grid_rank;
    MPI_Comm_rank(grid, &grid_rank);
    int coords[2];
    MPI_Cart_coords(grid, grid_rank, 2, coords);
    
    int my_row = coords[0];
    int my_col = coords[1];

    // Определяем соседей
    int up, down, left, right;
    MPI_Cart_shift(grid, 0, 1, &up, &down);   // Сдвиг по строкам (соседи сверху/снизу для B)
    MPI_Cart_shift(grid, 1, 1, &left, &right); // Сдвиг по столбцам (соседи слева/справа для A)

    // 3. Выделение памяти и инициализация
    int block = N / q;
    double *A = malloc(sizeof(double) * block * block);
    double *B = malloc(sizeof(double) * block * block);
    double *C = calloc(block * block, sizeof(double));

    // Глобальные индексы верхнего левого угла локального блока
    int row0 = my_row * block;
    int col0 = my_col * block;

    // --- ИЗМЕНЕНИЕ 1: Заполнение по заданию ---
    // A[i][j] = глобальный_i + глобальный_j
    // B[i][j] = 1.0 (заполнение единицами)
    for(int i = 0; i < block; i++){
        for(int j = 0; j < block; j++){
            int gi = row0 + i; // Глобальная строка
            int gj = col0 + j; // Глобальный столбец
            
            A[i*block + j] = (double)(gi + gj);
            B[i*block + j] = 1.0; 
        }
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 4. Начальное выравнивание (Skewing)
    // A сдвигаем влево на 'my_row' позиций
    // Примечание: цикл делает сдвиг на 1 шаг 'my_row' раз.
    for(int s = 0; s < my_row; s++){
        MPI_Sendrecv_replace(A, block*block, MPI_DOUBLE, left, 0, right, 0, grid, MPI_STATUS_IGNORE);
    }

    // B сдвигаем вверх на 'my_col' позиций
    for(int s = 0; s < my_col; s++){
        MPI_Sendrecv_replace(B, block*block, MPI_DOUBLE, up, 0, down, 0, grid, MPI_STATUS_IGNORE);
    }

    // 5. Основной цикл алгоритма Кэннона
    for(int step = 0; step < q; step++){
        // Локальное умножение C += A * B
        for(int i = 0; i < block; i++){
            for(int k = 0; k < block; k++){ 
                double a_val = A[i*block + k];
                for(int j = 0; j < block; j++){
                    C[i*block + j] += a_val * B[k*block + j];
                }
            }
        }

        // Циклический сдвиг блоков (A влево, B вверх)
        MPI_Sendrecv_replace(A, block*block, MPI_DOUBLE, left, 1, right, 1, grid, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B, block*block, MPI_DOUBLE, up, 2, down, 2, grid, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // --- ИЗМЕНЕНИЕ 2: Проверка результата ---
    // Теоретическое значение C[i][j] = Sum(A[i][k] * B[k][j]) по k=0..N-1
    // Так как B[k][j] = 1, то C[i][j] = Sum(i + k) = N*i + Sum(k)
    // Sum(k) от 0 до N-1 = (N-1)*N / 2
    
    long long global_err_count = 0;
    long long local_err_count = 0;
    double expected_sum_k = (double)N * (N - 1) / 2.0;
    double epsilon = 1e-5; // Погрешность для float сравнения

    for(int i = 0; i < block; i++){
        for(int j = 0; j < block; j++){
            int gi = row0 + i; // Глобальная строка текущего элемента C
            double expected = (double)N * gi + expected_sum_k;
            
            if (fabs(C[i*block + j] - expected) > epsilon) {
                local_err_count++;
            }
        }
    }
    
    MPI_Reduce(&local_err_count, &global_err_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank_world == 0){
        // Вывод времени в CSV формате
        printf("N,procs,time,status\n");
        if (global_err_count == 0) {
            printf("%d,%d,%.6f,OK\n", N, size, t1 - t0);
        } else {
            printf("%d,%d,%.6f,FAIL(errors=%lld)\n", N, size, t1 - t0, global_err_count);
        }
    }

    free(A); 
    free(B); 
    free(C);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}