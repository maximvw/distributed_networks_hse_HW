#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Гравитационная постоянная
#define G 6.67430e-11
// Шаг по времени
#define DT 0.01
// Размер блока CUDA (кол-во потоков в блоке)
#define BLOCK_SIZE 256

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Структура для удобного чтения на хосте (CPU)
typedef struct {
    double m;
    double x, y, z;
    double vx, vy, vz;
} ParticleHost;

// Ядро 1: Вычисление сил
// Каждый поток i считает сумму сил, действующих на i-ю частицу со стороны всех j
__global__ void compute_forces(int n, double *m, double *x, double *y, double *z, 
                               double *fx, double *fy, double *fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double my_x = x[i];
        double my_y = y[i];
        double my_z = z[i];
        double my_fx = 0.0;
        double my_fy = 0.0;
        double my_fz = 0.0;

        // Проходим по всем остальным частицам
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = x[j] - my_x;
            double dy = y[j] - my_y;
            double dz = z[j] - my_z;

            double dist_sq = dx*dx + dy*dy + dz*dz;
            // Softening parameter (как в исходном коде) для избежания деления на 0
            double dist = sqrt(dist_sq + 1e-10);
            double dist_cube = dist * dist * dist;

            // F = G * m1 * m2 / r^3 * vec(r)
            double f_mag = G * m[i] * m[j] / dist_cube;

            my_fx += f_mag * dx;
            my_fy += f_mag * dy;
            my_fz += f_mag * dz;
        }

        fx[i] = my_fx;
        fy[i] = my_fy;
        fz[i] = my_fz;
    }
}

// Ядро 2: Интеграция по Эйлеру
__global__ void integrate(int n, double *m, double *x, double *y, double *z, 
                          double *vx, double *vy, double *vz, 
                          double *fx, double *fy, double *fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double ax = fx[i] / m[i];
        double ay = fy[i] / m[i];
        double az = fz[i] / m[i];

        vx[i] += ax * DT;
        vy[i] += ay * DT;
        vz[i] += az * DT;

        x[i] += vx[i] * DT;
        y[i] += vy[i] * DT;
        z[i] += vz[i] * DT;
    }
}

int main(int argc, char *argv[]) {
    // Формат аргументов оставим совместимым с оригиналом, 
    // хотя nthreads для CUDA не играет роли (используем GPU грид).
    if (argc != 4) {
        printf("Usage: %s <nthreads_ignored> <tend> <filename>\n", argv[0]);
        return 1;
    }

    int threads_per_block = atoi(argv[1]); 
    double tend = atof(argv[2]);
    char *filename = argv[3];


    if (threads_per_block <= 0 || threads_per_block > 1024) {
        fprintf(stderr, "Warning: threads_per_block should be between 1 and 1024 for CUDA. Setting to 256.\n");
        threads_per_block = 256;
    }

    // 1. Чтение файла
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of particles\n");
        fclose(fp);
        return 1;
    }

    // Выделяем память на хосте для чтения
    ParticleHost *h_particles = (ParticleHost *)malloc(n * sizeof(ParticleHost));
    
    // Вспомогательные массивы на хосте для вывода (Structure of Arrays)
    double *h_x = (double*)malloc(n * sizeof(double));
    double *h_y = (double*)malloc(n * sizeof(double));
    double *h_z = (double*)malloc(n * sizeof(double));
    double *h_m = (double*)malloc(n * sizeof(double));
    double *h_vx = (double*)malloc(n * sizeof(double));
    double *h_vy = (double*)malloc(n * sizeof(double));
    double *h_vz = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        // Читаем в структуру (как в исходном коде)
        int read_count = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", 
            &h_particles[i].m, 
            &h_particles[i].x, &h_particles[i].y, &h_particles[i].z, 
            &h_particles[i].vx, &h_particles[i].vy, &h_particles[i].vz);
            
        if (read_count != 7) {
            fprintf(stderr, "Error reading particle %d.\n", i);
            free(h_particles); fclose(fp); return 1;
        }

        // Перекладываем в плоские массивы для отправки на GPU
        h_m[i] = h_particles[i].m;
        h_x[i] = h_particles[i].x; h_y[i] = h_particles[i].y; h_z[i] = h_particles[i].z;
        h_vx[i] = h_particles[i].vx; h_vy[i] = h_particles[i].vy; h_vz[i] = h_particles[i].vz;
    }
    fclose(fp);
    free(h_particles); // Структуры больше не нужны

    // 2. Выделение памяти на GPU (Device)
    double *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
    size_t bytes = n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));
    CUDA_CHECK(cudaMalloc(&d_fx, bytes));
    CUDA_CHECK(cudaMalloc(&d_fy, bytes));
    CUDA_CHECK(cudaMalloc(&d_fz, bytes));

    // 3. Копирование данных Host -> Device
    CUDA_CHECK(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice));

    // Настройка сетки запуска (Grid/Block)
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Тайминг через CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int steps = (int)(tend / DT);

    // Вывод начального состояния (t=0)
    printf("0.000000");
    for (int i = 0; i < n; i++) {
        printf(",%f,%f,%f", h_x[i], h_y[i], h_z[i]);
    }
    printf("\n");

    // Основной цикл
    for (int s = 1; s <= steps; s++) {
        double current_time = s * DT;

        // 1. Расчет сил
        compute_forces<<<blocks, threads_per_block>>>(n, d_m, d_x, d_y, d_z, d_fx, d_fy, d_fz);
        CUDA_CHECK(cudaGetLastError());

        // 2. Интеграция
        integrate<<<blocks, threads_per_block>>>(n, d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
        CUDA_CHECK(cudaGetLastError());

        // 3. Копирование координат обратно для вывода (это "узкое место", но нужно для формата CSV)
        CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));
        
        // Синхронизация не обязательна явно перед printf, так как cudaMemcpy блокирующий

        // 4. Вывод
        printf("%f", current_time);
        for (int i = 0; i < n; i++) {
            printf(",%f,%f,%f", h_x[i], h_y[i], h_z[i]);
        }
        printf("\n");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Вывод времени в stderr, чтобы не портить CSV
    fprintf(stderr, "Time taken: %.4f seconds\n", milliseconds / 1000.0);

    // Освобождение памяти
    free(h_x); free(h_y); free(h_z); free(h_m);
    free(h_vx); free(h_vy); free(h_vz);
    
    cudaFree(d_m);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);

    return 0;
}