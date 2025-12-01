#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Максимальное количество итераций для проверки сходимости
// Чем больше число, тем точнее границы множества, но дольше расчет.
#define MAX_ITER 2000

// Границы области на комплексной плоскости, где мы ищем множество

const double MIN_X = -3.0;
const double MAX_X = 3.0;
const double MIN_Y = -3.0;
const double MAX_Y = 3.0;

int main(int argc, char *argv[]) {
    // 1. Проверка аргументов командной строки
    if (argc != 3) {
        printf("Usage: %s nthreads npoints\n", argv[0]);
        return 1;
    }

    int nthreads = atoi(argv[1]); // Количество потоков
    long long npoints = atoll(argv[2]); // Общее количество точек

    if (npoints <= 0 || nthreads <= 0) {
        printf("Error: nthreads and npoints must be positive integers.\n");
        return 1;
    }

    // Устанавливаем количество потоков OpenMP
    omp_set_num_threads(nthreads);

    // 2. Подготовка сетки
    // Мы хотим проверить примерно npoints точек.
    // Представим их как квадратную сетку side x side.
    long side = (long)sqrt((double)npoints);
    
    // Шаг сетки по координатам
    double dx = (MAX_X - MIN_X) / side;
    double dy = (MAX_Y - MIN_Y) / side;

    printf("Computing Mandelbrot set for approx %ld points (%ld x %ld grid)...\n", side*side, side, side);
    printf("Threads: %d\n", nthreads);

    // Открываем файл для записи результатов
    FILE *fp = fopen("results/task_results/mandelbrot.csv", "w");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }
    
    // Записываем заголовок CSV (опционально, но полезно)
    fprintf(fp, "x,y\n");

    // Замеряем время (для отчета)
    double start_time = omp_get_wtime();

    // 3. Основной параллельный цикл
    // Используем collapse(2), чтобы объединить вложенные циклы для лучшего распараллеливания
    // schedule(dynamic) помогает сбалансировать нагрузку, так как некоторые точки вылетают быстро, а другие долго крутятся в цикле
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long i = 0; i < side; i++) {
        for (long j = 0; j < side; j++) {
            
            // Вычисляем координаты точки c = x + iy
            double x0 = MIN_X + i * dx;
            double y0 = MIN_Y + j * dy;

            // Начальные значения z = 0
            double z_re = 0.0;
            double z_im = 0.0;
            
            // Переменная для проверки принадлежности множеству
            int in_set = 1; // Предполагаем, что точка в множестве

            // Итерационный процесс: z = z^2 + c
            for (int k = 0; k < MAX_ITER; k++) {
                // z^2 = (re + i*im)^2 = re^2 - im^2 + 2*i*re*im
                double z_re2 = z_re * z_re;
                double z_im2 = z_im * z_im;

                // Проверка условия выхода: |z| >= 2, то есть re^2 + im^2 >= 4
                if (z_re2 + z_im2 > 4.0) {
                    in_set = 0; // Точка улетела в бесконечность
                    break;
                }

                // Пересчитываем z
                // Новая мнимая часть: 2 * re * im + y0
                z_im = 2.0 * z_re * z_im + y0;
                // Новая реальная часть: re^2 - im^2 + x0
                z_re = z_re2 - z_im2 + x0;
            }

            // 4. Запись результата
            if (in_set) {
                // ВАЖНО: Запись в файл должна быть последовательной (thread-safe).
                // Используем критическую секцию.
                // Примечание: Для огромного количества точек это может быть узким местом.
                // Но для лабораторной работы это самый простой и надежный способ.
                #pragma omp critical
                {
                    fprintf(fp, "%.6f,%.6f\n", x0, y0);
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    fclose(fp);

    printf("Done. Results saved to mandelbrot.csv\n");
    printf("Time taken: %.4f seconds\n", end_time - start_time);

    return 0;
}