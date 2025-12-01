#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Функция источника тепла f(x, y)
double func(double x, double y) {
    // Пример: источник в центре области (0.5, 0.5) радиусом 0.1
    double dx = x - 0.5;
    double dy = y - 0.5;
    if (dx*dx + dy*dy < 0.1) return 10.0; // Сильный источник тепла
    return 0.0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1000;      // Размер сетки (внутренних точек)
    int ITMAX = (argc > 2) ? atoi(argv[2]) : 100;   // Количество итераций
    double u_boundary = 10.0;                       // Температура на краях


    // 1. Топология
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int px = dims[0];
    int py = dims[1]; 
    
    int periods[2] = {0, 0};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);
    
    int grid_rank;
    MPI_Comm_rank(grid, &grid_rank);
    int coords[2];
    MPI_Cart_coords(grid, grid_rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    int up, down, left, right;
    MPI_Cart_shift(grid, 0, 1, &up, &down);
    MPI_Cart_shift(grid, 1, 1, &left, &right);

    // 2. Декомпозиция
    // Вычисляем размеры локального блока (строки - nx, столбцы - ny в твоей нотации)
    // Примечание: обычно nx - это X (столбцы), ny - Y (строки), но у тебя nx используется как размер первой размерности массива (строки).
    // Чтобы не путаться, оставим твои переменные, но уточним смысл:
    // nx - количество локальных строк (dim 0), ny - количество локальных столбцов (dim 1)
    
    int base_rows = N / px;
    int base_cols = N / py;
    
    int local_rows = base_rows + (my_row < N % px ? 1 : 0);
    int local_cols = base_cols + (my_col < N % py ? 1 : 0);

    // !!! ВАЖНО: Вычисляем глобальное смещение для координат !!!
    int offset_row = 0;
    for (int k = 0; k < my_row; k++) {
        offset_row += base_rows + (k < N % px ? 1 : 0);
    }
    
    int offset_col = 0;
    for (int k = 0; k < my_col; k++){ 
        offset_col += base_cols + (k < N % py ? 1 : 0);
    }

    // u[строка][столбец] -> u[local_rows+2][local_cols+2]
    double *u = calloc((local_rows + 2) * (local_cols + 2), sizeof(double));
    
    // Шаг сетки (общий для всей области 1.0 x 1.0)
    double h = 1.0 / (N + 1);

    // Инициализация границ
    // Верхняя граница глобальной области
    if (up == MPI_PROC_NULL) {
        for (int j = 0; j <= local_cols + 1; j++){
            u[0 * (local_cols + 2) + j] = u_boundary;
        }
    }
    // Нижняя граница глобальной области
    if (down == MPI_PROC_NULL) {
        for (int j = 0; j <= local_cols + 1; j++){
            u[(local_rows + 1) * (local_cols + 2) + j] = u_boundary;
        }
    }
    // Левая граница
    if (left == MPI_PROC_NULL) {
        for (int i = 0; i <= local_rows + 1; i++){
            u[i * (local_cols + 2) + 0] = u_boundary;
        }
    }
    // Правая граница
    if (right == MPI_PROC_NULL) {
        for (int i = 0; i <= local_rows + 1; i++){
            u[i * (local_cols + 2) + (local_cols + 1)] = u_boundary;
        }
    }


    // Буферы
    double *send_row_first = malloc(local_cols * sizeof(double));
    double *send_row_last  = malloc(local_cols * sizeof(double));
    double *recv_row_up    = malloc(local_cols * sizeof(double));
    double *recv_row_down  = malloc(local_cols * sizeof(double));

    double *send_col_first = malloc(local_rows * sizeof(double));
    double *send_col_last  = malloc(local_rows * sizeof(double));
    double *recv_col_left  = malloc(local_rows * sizeof(double));
    double *recv_col_right = malloc(local_rows * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 3. Волновая схема
    for (int it = 0; it < ITMAX; it++) {
        
        // --- ФАЗА 1: Получение фронта волны (данные итерации k+1 от соседей сверху и слева) ---
        if (up != MPI_PROC_NULL) {
            MPI_Recv(recv_row_up, local_cols, MPI_DOUBLE, up, 0, grid, MPI_STATUS_IGNORE);
            for (int j = 1; j <= local_cols; j++){
                u[0 * (local_cols + 2) + j] = recv_row_up[j-1];
            }
        }
        
        if (left != MPI_PROC_NULL) {
            MPI_Recv(recv_col_left, local_rows, MPI_DOUBLE, left, 1, grid, MPI_STATUS_IGNORE);
            for (int i = 1; i <= local_rows; i++){
                u[i * (local_cols + 2) + 0] = recv_col_left[i-1];
            }
        }

        // --- ФАЗА 2: Вычисления ---
        double max_diff = 0.0;
        
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                // !!! ИСПРАВЛЕНИЕ: Глобальные координаты !!!
                // i - локальный индекс строки (1..local_rows)
                // j - локальный индекс столбца (1..local_cols)
                // Глобальный индекс: Offset + (i-1) + 1 (т.к. сетка с 1 до N)
                double gx = (offset_col + j) * h; // x идет по столбцам
                double gy = (offset_row + i) * h; // y идет по строкам (обычно в матрицах так, либо наоборот, зависит от условий)
                // В условии задачи: u(x,y). Обычно x - горизонталь (j), y - вертикаль (i).
                // Уточним: d2u/dx2 + d2u/dy2. Порядок не важен для формулы, но важен для func.
                
                double val_f = func(gx, gy); 
                
                double old_val = u[i * (local_cols + 2) + j];
                double new_val = 0.25 * (
                    u[(i - 1) * (local_cols + 2) + j] + // Верх
                    u[(i + 1) * (local_cols + 2) + j] + // Низ
                    u[i * (local_cols + 2) + (j - 1)] + // Лево
                    u[i * (local_cols + 2) + (j + 1)] - // Право
                    h * h * val_f
                );
                u[i * (local_cols + 2) + j] = new_val;
                
                double diff = fabs(new_val - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }

        // --- ФАЗА 3: Распространение волны (отправка k+1 вниз и вправо) ---
        if (down != MPI_PROC_NULL) {
            for (int j = 1; j <= local_cols; j++) send_row_last[j-1] = u[local_rows * (local_cols + 2) + j];
            MPI_Send(send_row_last, local_cols, MPI_DOUBLE, down, 0, grid);
        }
        
        if (right != MPI_PROC_NULL) {
            for (int i = 1; i <= local_rows; i++) send_col_last[i-1] = u[i * (local_cols + 2) + local_cols];
            MPI_Send(send_col_last, local_rows, MPI_DOUBLE, right, 1, grid);
        }

        // --- ФАЗА 4: Обновление границ для следующей итерации (данные k от соседей снизу и справа) ---
        // Обмен Снизу-Вверх: Мы отправляем Вверх нашу первую строку, получаем от Низа его первую строку (в наш нижний гало)
        
        // Подготовка данных для отправки Вверх (наша первая строка)
        for(int j=1; j<=local_cols; j++) send_row_first[j-1] = u[1 * (local_cols + 2) + j];

        MPI_Sendrecv(send_row_first, local_cols, MPI_DOUBLE, up, 2,     
                     recv_row_down, local_cols, MPI_DOUBLE, down, 2,    
                     grid, MPI_STATUS_IGNORE);
                     
        if (down != MPI_PROC_NULL)
             for (int j=1; j<=local_cols; j++) u[(local_rows+1)*(local_cols+2)+j] = recv_row_down[j-1];

        // Обмен Справа-Влево: Отправляем Влево наш первый столбец, получаем от Правого его первый столбец (в наш правый гало)
        for(int i=1; i<=local_rows; i++) send_col_first[i-1] = u[i*(local_cols+2) + 1];
        
        MPI_Sendrecv(send_col_first, local_rows, MPI_DOUBLE, left, 3,
                     recv_col_right, local_rows, MPI_DOUBLE, right, 3,
                     grid, MPI_STATUS_IGNORE);
                     
        if (right != MPI_PROC_NULL)
             for (int i=1; i<=local_rows; i++) u[i*(local_cols+2) + local_cols+1] = recv_col_right[i-1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double total_time;
    double local_time = t1 - t0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (grid_rank == 0) {
        printf("N,procs,px,py,itmax,time\n");
        printf("%d,%d,%d,%d,%d,%.6f\n", N, size, px, py, ITMAX, total_time);
    }

    // Очистка памяти
    free(u);
    free(send_row_first); free(send_row_last); free(recv_row_up); free(recv_row_down);
    free(send_col_first); free(send_col_last); free(recv_col_left); free(recv_col_right);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}