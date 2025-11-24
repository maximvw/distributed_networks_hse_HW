#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Функция для заполнения массива случайными числами
void fill_random(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверка аргументов (ожидается размер N)
    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);

    // 1. Создание декартовой топологии (решетки процессов)
    int dims[2] = {0, 0}; // [rows, cols]
    MPI_Dims_create(size, 2, dims);
    
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // Получение координат текущего процесса в решетке
    int coords[2];
    int my_rank_cart;
    MPI_Comm_rank(cart_comm, &my_rank_cart); // Ранг может измениться после reorder
    MPI_Cart_coords(cart_comm, my_rank_cart, 2, coords);
    int grid_row = coords[0];
    int grid_col = coords[1];

    // 2. Определение размеров локального блока
    // Распределение строк матрицы по строкам решетки процессов
    int base_rows = N / dims[0];
    int rem_rows = N % dims[0];
    int my_rows = base_rows + (grid_row < rem_rows ? 1 : 0);

    // Распределение столбцов матрицы по столбцам решетки процессов
    int base_cols = N / dims[1];
    int rem_cols = N % dims[1];
    int my_cols = base_cols + (grid_col < rem_cols ? 1 : 0);

    // 3. Выделение памяти
    // Локальная часть матрицы A (блок)
    double* local_A = (double*)malloc(my_rows * my_cols * sizeof(double));
    // Локальная часть вектора b (соответствует столбцам блока)
    double* local_b = (double*)malloc(my_cols * sizeof(double));
    // Локальная часть результата c (соответствует строкам блока)
    double* local_c = (double*)malloc(my_rows * sizeof(double));
    // Буфер для финального сбора результата (нужен только корням строк)
    double* final_c_part = (double*)malloc(my_rows * sizeof(double));

    // Инициализация данных (заполняем случайными числами)
    // Для измерения производительности достаточно заполнить локально
    srand(time(NULL) + rank);
    fill_random(local_A, my_rows * my_cols);
    
    // Логика инициализации вектора b:
    // В реальной задаче вектор распределен.
    // Пусть процессы первой строки решетки (grid_row == 0) генерируют вектор b.
    if (grid_row == 0) {
        fill_random(local_b, my_cols);
    } else {
        // Остальные обнуляют, чтобы потом принять данные
        for(int i=0; i<my_cols; i++) local_b[i] = 0.0;
    }

    // Инициализируем результат нулями
    for (int i = 0; i < my_rows; i++) local_c[i] = 0.0;

    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // 4. Коммуникация: Рассылка вектора b по столбцам решетки
    // Создаем коммуникатор для столбца решетки
    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, grid_col, grid_row, &col_comm);

    // Процесс с grid_row == 0 рассылает свой кусок вектора b всем в своем столбце
    MPI_Bcast(local_b, my_cols, MPI_DOUBLE, 0, col_comm);

    // 5. Вычисления: Локальное умножение матрицы на вектор
    // c = A * b
    for (int i = 0; i < my_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < my_cols; j++) {
            sum += local_A[i * my_cols + j] * local_b[j];
        }
        local_c[i] = sum;
    }

    // 6. Коммуникация: Сбор результатов по строкам решетки
    // Нам нужно сложить частичные суммы local_c от всех процессов в одной строке решетки
    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, grid_row, grid_col, &row_comm);

    // Редукция суммы в процесс с grid_col == 0
    MPI_Reduce(local_c, final_c_part, my_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();

    // 7. Вывод результатов
    // Формат: N,procs,time,px,py
    // Выводит только Rank 0 глобального коммуникатора
    if (rank == 0) {
        printf("%d,%d,%.6f,%d,%d\n", N, size, end_time - start_time, dims[0], dims[1]);
    }

    // Очистка памяти
    free(local_A);
    free(local_b);
    free(local_c);
    free(final_c_part);
    
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}