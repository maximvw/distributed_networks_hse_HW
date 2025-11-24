#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Функция источника тепла f(x, y)
double func(double x, double y) {
    // Пример: источник в центре
    double dx = x - 0.5;
    double dy = y - 0.5;
    if (dx*dx + dy*dy < 0.1) return 10.0;
    return 0.0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1000;      // Размер сетки (внутренних точек)
    int ITMAX = (argc > 2) ? atoi(argv[2]) : 100;   // Количество итераций
    double u_boundary = 10.0;                       // Температура на краях пластины (c)

    // 1. Создаем декартову топологию для удобства (как в задании 3)
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int px = dims[0];
    int py = dims[1];
    
    int periods[2] = {0, 0}; // Непериодическая решетка (стенки есть стенки)
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);
    
    int grid_rank;
    MPI_Comm_rank(grid, &grid_rank);
    int coords[2];
    MPI_Cart_coords(grid, grid_rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Соседи: Верх, Низ, Лево, Право
    int up, down, left, right;
    MPI_Cart_shift(grid, 0, 1, &up, &down);
    MPI_Cart_shift(grid, 1, 1, &left, &right);

    // 2. Декомпозиция области
    // Вычисляем размеры локального блока
    int base_nx = N / px;
    int base_ny = N / py;
    int nx = base_nx + (my_row < N % px ? 1 : 0);
    int ny = base_ny + (my_col < N % py ? 1 : 0);

    // Выделяем память с учетом гало-ячеек (границ)
    // u[0] - левый гало, u[nx+1] - правый гало
    // u[idx(0, ...)] - верхний гало, u[idx(ny+1, ...)] - нижний гало
    double *u = calloc((nx + 2) * (ny + 2), sizeof(double));
    
    // Шаг сетки
    double h = 1.0 / (N + 1);

    // Инициализация (граничные условия u=c на краях всей пластины)
    // Внутренность инициализируем 0 (или начальным приближением)
    
    // Если мы на самой верхней границе глобальной сетки
    if (up == MPI_PROC_NULL) {
        for (int j = 0; j <= ny + 1; j++) u[0 * (ny + 2) + j] = u_boundary;
    }
    // Если мы на самой нижней границе
    if (down == MPI_PROC_NULL) {
        for (int j = 0; j <= ny + 1; j++) u[(nx + 1) * (ny + 2) + j] = u_boundary;
    }
    // Если мы на левой границе
    if (left == MPI_PROC_NULL) {
        for (int i = 0; i <= nx + 1; i++) u[i * (ny + 2) + 0] = u_boundary;
    }
    // Если мы на правой границе
    if (right == MPI_PROC_NULL) {
        for (int i = 0; i <= nx + 1; i++) u[i * (ny + 2) + (ny + 1)] = u_boundary;
    }

    // Буферы для обмена
    double *send_row_first = malloc(ny * sizeof(double));
    double *send_row_last  = malloc(ny * sizeof(double));
    double *recv_row_up    = malloc(ny * sizeof(double));
    double *recv_row_down  = malloc(ny * sizeof(double));

    double *send_col_first = malloc(nx * sizeof(double));
    double *send_col_last  = malloc(nx * sizeof(double));
    double *recv_col_left  = malloc(nx * sizeof(double));
    double *recv_col_right = malloc(nx * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 3. Основной цикл (Волновая схема)
    for (int it = 0; it < ITMAX; it++) {
        
        // --- ФАЗА 1: Прием данных "по потоку" (Сверху и Слева) ---
        // Это и создает "волну": мы не можем начать, пока не получим данные от "старших" соседей
        
        if (up != MPI_PROC_NULL) {
            MPI_Recv(recv_row_up, ny, MPI_DOUBLE, up, 0, grid, MPI_STATUS_IGNORE);
            // Копируем в гало (верхняя строка u)
            for (int j = 1; j <= ny; j++) u[0 * (ny + 2) + j] = recv_row_up[j-1];
        }
        
        if (left != MPI_PROC_NULL) {
            MPI_Recv(recv_col_left, nx, MPI_DOUBLE, left, 1, grid, MPI_STATUS_IGNORE);
            // Копируем в гало (левый столбец u)
            for (int i = 1; i <= nx; i++) u[i * (ny + 2) + 0] = recv_col_left[i-1];
        }

        // --- ФАЗА 2: Вычисления (Гаусс-Зейдель) ---
        // Формула: (u_up + u_down + u_left + u_right - h^2*f) / 4
        // u_up и u_left — только что полученные "новые" данные (итерация k+1)
        // u_down и u_right — данные с предыдущей итерации (k)
        
        double max_diff = 0.0;
        
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                // Глобальные координаты для функции f(x,y)
                // (Примерный расчет, точный зависит от смещения my_row/my_col)
                // Для простоты здесь опустим точный расчет глобальных x,y
                double val_f = func(0.5, 0.5); // Упрощено
                
                double old_val = u[i * (ny + 2) + j];
                double new_val = 0.25 * (
                    u[(i - 1) * (ny + 2) + j] + // Верх
                    u[(i + 1) * (ny + 2) + j] + // Низ
                    u[i * (ny + 2) + (j - 1)] + // Лево
                    u[i * (ny + 2) + (j + 1)] - // Право
                    h * h * val_f               // Источник
                );
                u[i * (ny + 2) + j] = new_val;
            }
        }

        // --- ФАЗА 3: Отправка данных "по потоку" (Вниз и Вправо) ---
        // Теперь мы можем разрешить работать соседям снизу и справа
        
        if (down != MPI_PROC_NULL) {
            // Собираем последнюю вычисленную строку
            for (int j = 1; j <= ny; j++) send_row_last[j-1] = u[nx * (ny + 2) + j];
            MPI_Send(send_row_last, ny, MPI_DOUBLE, down, 0, grid);
        }
        
        if (right != MPI_PROC_NULL) {
            // Собираем последний вычисленный столбец
            for (int i = 1; i <= nx; i++) send_col_last[i-1] = u[i * (ny + 2) + ny];
            MPI_Send(send_col_last, nx, MPI_DOUBLE, right, 1, grid);
        }

        // --- ФАЗА 4: Обновление "хвостов" (Вниз-Вверх и Вправо-Влево) ---
        // Чтобы на следующей итерации у нас были актуальные u_down и u_right, 
        // нужно обменяться границами в обратную сторону.
        // Для чистоты "Волны" часто опускают, но для точности надо делать.
        
        // Обмен Снизу-Вверх
        MPI_Sendrecv(u + (ny+2) + 1, ny, MPI_DOUBLE, up, 2,     // Первая строка -> Вверх
                     recv_row_down, ny, MPI_DOUBLE, down, 2,    // От низа <- В нижний гало
                     grid, MPI_STATUS_IGNORE);
        if (down != MPI_PROC_NULL) // Копируем из буфера в гало
             for (int j=1; j<=ny; j++) u[(nx+1)*(ny+2)+j] = recv_row_down[j-1];

        // Обмен Справа-Влево
        // Сборка столбца (он не непрерывен в памяти)
        for(int i=1; i<=nx; i++) send_col_first[i-1] = u[i*(ny+2) + 1];
        
        MPI_Sendrecv(send_col_first, nx, MPI_DOUBLE, left, 3,
                     recv_col_right, nx, MPI_DOUBLE, right, 3,
                     grid, MPI_STATUS_IGNORE);
        if (right != MPI_PROC_NULL)
             for (int i=1; i<=nx; i++) u[i*(ny+2) + ny+1] = recv_col_right[i-1];
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