#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int N = atoi(argv[1]);

    // 1. Распределение столбцов
    // Каждому процессу достается набор столбцов
    int cols_per = N / size;
    int rem = N % size;
    
    // Определяем начальный глобальный индекс столбца и количество столбцов для текущего процесса
    int my_first = rank * cols_per + (rank < rem ? rank : rem);
    int my_cols = cols_per + (rank < rem ? 1 : 0);

    // 2. Выделение памяти
    // Локальная часть матрицы A: N строк, my_cols столбцов
    // Храним в одномерном массиве, доступ: row * width + col
    double *A = malloc(sizeof(double) * N * my_cols);
    double *x_local = malloc(sizeof(double) * my_cols);
    
    // Частичная сумма результата (каждый процесс считает вклад своих столбцов в полный вектор Y)
    double *y_local = malloc(sizeof(double) * N); 
    
    // Итоговый результат (нужен только на root, но выделим для reduce)
    double *y = NULL;
    if (rank == 0) {
        y = malloc(sizeof(double) * N);
    }

    // 3. Инициализация данных по формуле
    // Матрица A[i][j] = i + j
    // Вектор x[j] = 1
    for (int j = 0; j < my_cols; j++) {
        int global_j = my_first + j; // Глобальный индекс столбца
        x_local[j] = 1.0; 
        
        for (int i = 0; i < N; i++) {
            // A хранится локально как N строк по my_cols элементов
            A[i * my_cols + j] = (double)(i + global_j);
        }
    }

    // Обнуляем локальный вектор результата перед накоплением суммы
    for (int i = 0; i < N; i++) {
        y_local[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 4. Вычисления (Матрица * Вектор)
    // Каждый процесс умножает свои столбцы матрицы на свою часть вектора
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < my_cols; j++) {
            sum += A[i * my_cols + j] * x_local[j];
        }
        y_local[i] = sum;
    }

    // 5. Сбор результатов
    // Поскольку разбиение по столбцам, каждый процесс получил частичные суммы для ВСЕГО вектора Y.
    // Нам нужно просто сложить y_local всех процессов.
    MPI_Reduce(y_local, y, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // 6. Проверка результата (только на Rank 0)
    if (rank == 0) {
        int errors = 0;

        double arithmetic_sum_j = (double)N * (N - 1) / 2.0;

        for (int i = 0; i < N; i++) {
            double expected = (double)N * i + arithmetic_sum_j;
            if (fabs(y[i] - expected) > 1e-1) { // 1e-1 для грубой защиты от float погрешностей на больших N
                errors++;

                printf("Error at index %d: expected %.2f, got %.2f\n", i, expected, y[i]);
                if (errors > 10) break; 
            }
        }

        if (errors == 0) {
            printf("OK, Size=%d, Procs=%d, Time=%.6f\n", N, size, t1 - t0);
        } else {
            printf("FAIL, Size=%d, Procs=%d, Errors=%d\n", N, size, errors);
        }
    }

    // Очистка памяти
    free(A);
    free(x_local);
    free(y_local);
    if (rank == 0) {
        free(y);
    }
    
    MPI_Finalize();
    return 0;
}