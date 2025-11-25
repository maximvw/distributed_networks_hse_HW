#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Вспомогательная функция для расчета глобального смещения
// grid_coord - координата процесса в решетке (строка или столбец)
// dims_len - размер решетки по этому измерению
// N - общий размер задачи
int get_global_offset(int grid_coord, int dims_len, int N) {
    int base = N / dims_len;
    int rem = N % dims_len;
    // Смещение = (количество полных блоков * базовый размер) + (добавочные строки для первых rem процессов)
    if (grid_coord < rem) {
        return grid_coord * (base + 1);
    } else {
        return rem * (base + 1) + (grid_coord - rem) * base;
    }
}

int main(int argc, char** argv) {
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

    // 1. Топология
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2], my_rank_cart;
    MPI_Comm_rank(cart_comm, &my_rank_cart);
    MPI_Cart_coords(cart_comm, my_rank_cart, 2, coords);
    int grid_row = coords[0];
    int grid_col = coords[1];

    // 2. Расчет размеров и смещений
    // Строки (rows)
    int base_rows = N / dims[0];
    int rem_rows = N % dims[0];
    int my_rows = base_rows + (grid_row < rem_rows ? 1 : 0);
    int global_row_offset = get_global_offset(grid_row, dims[0], N);

    // Столбцы (cols)
    int base_cols = N / dims[1];
    int rem_cols = N % dims[1];
    int my_cols = base_cols + (grid_col < rem_cols ? 1 : 0);
    int global_col_offset = get_global_offset(grid_col, dims[1], N);

    // 3. Выделение памяти
    double* local_A = (double*)malloc(my_rows * my_cols * sizeof(double));
    double* local_b = (double*)malloc(my_cols * sizeof(double));
    double* local_c = (double*)malloc(my_rows * sizeof(double));
    double* final_c_part = (double*)malloc(my_rows * sizeof(double));

    // --- ИНИЦИАЛИЗАЦИЯ ПО ФОРМУЛЕ ---
    // A[i][j] = global_i + global_j
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < my_cols; j++) {
            int global_i = global_row_offset + i;
            int global_j = global_col_offset + j;
            local_A[i * my_cols + j] = (double)(global_i + global_j);
        }
    }

    // b[j] = 1.0 (заполняет только верхняя строка процессов)
    if (grid_row == 0) {
        for (int j = 0; j < my_cols; j++) {
            local_b[j] = 1.0;
        }
    } else {
        for (int j = 0; j < my_cols; j++) local_b[j] = 0.0;
    }

    for (int i = 0; i < my_rows; i++) local_c[i] = 0.0;

    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // 4. Рассылка вектора b по столбцам
    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, grid_col, grid_row, &col_comm);
    MPI_Bcast(local_b, my_cols, MPI_DOUBLE, 0, col_comm);

    // 5. Умножение
    for (int i = 0; i < my_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < my_cols; j++) {
            sum += local_A[i * my_cols + j] * local_b[j];
        }
        local_c[i] = sum;
    }

    // 6. Сбор (Редукция по строкам)
    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, grid_row, grid_col, &row_comm);
    // Суммируем частичные суммы local_c в final_c_part на корне строки (grid_col == 0)
    MPI_Reduce(local_c, final_c_part, my_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();

    // 7. Проверка результата (только на процессах первого столбца)
    int local_errors = 0;
    if (grid_col == 0) {
        for (int i = 0; i < my_rows; i++) {
            int global_i = global_row_offset + i;
            // Ожидаемое значение: N*i + N*(N-1)/2
            double expected = (double)N * global_i + (double)N * (N - 1) / 2.0;
            
            // Сравниваем с небольшой погрешностью
            if (fabs(final_c_part[i] - expected) > 1e-1) {
                local_errors++;
            }
        }
    }

    // Собираем ошибки со всех процессов
    int total_errors = 0;
    MPI_Reduce(&local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 8. Вывод (парсится автотестом)
    if (rank == 0) {
        if (total_errors == 0) {
            printf("OK,%d,%d,%.6f\n", N, size, end_time - start_time);
        } else {
            printf("FAIL,%d,%d,%d\n", N, size, total_errors);
        }
    }

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