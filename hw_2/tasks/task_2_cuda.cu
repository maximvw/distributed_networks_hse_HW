#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Гравитационная постоянная
#define G 6.67430e-11f
// Шаг по времени (Delta t) - можно подобрать под задачу
#define DT 0.01f
// Количество потоков в блоке
#define BLOCK_SIZE 256

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// CUDA ядро для шага симуляции (Метод Эйлера)
__global__ void nbody_step(
    int n, 
    const float *m, 
    const float *x, const float *y, const float *z,
    const float *vx, const float *vy, const float *vz,
    float *new_x, float *new_y, float *new_z,
    float *new_vx, float *new_vy, float *new_vz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    
    // Текущее положение тела i
    float rx_i = x[i];
    float ry_i = y[i];
    float rz_i = z[i];
    float mass_i = m[i];

    // Вычисляем силы притяжения со стороны всех остальных тел
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = x[j] - rx_i;
        float dy = y[j] - ry_i;
        float dz = z[j] - rz_i;
        
        // Квадрат расстояния + epsilon для избежания деления на 0 (softening)
        float dist_sq = dx*dx + dy*dy + dz*dz + 1e-9f;
        float dist = sqrtf(dist_sq);
        float dist_cube = dist_sq * dist;

        // Закон всемирного тяготения: F = G * m1 * m2 / r^3 * vec(r)
        float f = (G * mass_i * m[j]) / dist_cube;

        fx += f * dx;
        fy += f * dy;
        fz += f * dz;
    }

    // Метод Эйлера
    // a = F / m
    float ax = fx / mass_i;
    float ay = fy / mass_i;
    float az = fz / mass_i;

    // v_new = v_old + a * dt
    float nvx = vx[i] + ax * DT;
    float nvy = vy[i] + ay * DT;
    float nvz = vz[i] + az * DT;

    // r_new = r_old + v_old * dt (согласно формуле (8) в PDF используется старая скорость v^{n-1})
    float nx = rx_i + vx[i] * DT;
    float ny = ry_i + vy[i] * DT;
    float nz = rz_i + vz[i] * DT;

    // Запись в буферы "следующего" шага
    new_vx[i] = nvx;
    new_vy[i] = nvy;
    new_vz[i] = nvz;
    new_x[i] = nx;
    new_y[i] = ny;
    new_z[i] = nz;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <t_end> <filename>\n", argv[0]);
        return 1;
    }

    float t_end = atof(argv[1]);
    char *filename = argv[2];

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of bodies\n");
        return 1;
    }

    // Выделение памяти на хосте (CPU)
    size_t bytes = n * sizeof(float);
    float *h_m = (float*)malloc(bytes);
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_z = (float*)malloc(bytes);
    float *h_vx = (float*)malloc(bytes);
    float *h_vy = (float*)malloc(bytes);
    float *h_vz = (float*)malloc(bytes);

    // Чтение файла. Ожидаемый формат строки: m x y z vx vy vz
    for (int i = 0; i < n; i++) {
        // Если в файле нет массы (только 6 чисел), замените строку ниже на:
        // h_m[i] = 1.0f; fscanf(fp, "%f %f %f %f %f %f", ...
        if (fscanf(fp, "%f %f %f %f %f %f %f", 
            &h_m[i], 
            &h_x[i], &h_y[i], &h_z[i], 
            &h_vx[i], &h_vy[i], &h_vz[i]) != 7) {
            
            fprintf(stderr, "Error reading body %d data\n", i);
            // Попытка fallback если массы нет (только 6 чисел)
            // h_m[i] = 1.0f; // раскомментировать при необходимости
        }
    }
    fclose(fp);

    // Выделение памяти на устройстве (GPU)
    // Используем двойную буферизацию для координат и скоростей, чтобы избежать гонки данных
    float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    float *d_new_x, *d_new_y, *d_new_z, *d_new_vx, *d_new_vy, *d_new_vz;

    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));
    
    CUDA_CHECK(cudaMalloc(&d_new_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_new_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_new_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_new_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_new_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_new_vz, bytes));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float current_time = 0.0f;

    // Вывод начального состояния (t=0)
    printf("t");
    for(int k=0; k<n; k++) printf(", x%d, y%d, z%d", k+1, k+1, k+1);
    printf("\n");
    
    printf("%.4f", current_time);
    for (int i = 0; i < n; i++) {
        printf(", %f, %f, %f", h_x[i], h_y[i], h_z[i]);
    }
    printf("\n");

    // Основной цикл по времени
    while (current_time < t_end) {
        // Запуск ядра
        nbody_step<<<blocks, BLOCK_SIZE>>>(
            n, d_m, 
            d_x, d_y, d_z, d_vx, d_vy, d_vz,
            d_new_x, d_new_y, d_new_z, d_new_vx, d_new_vy, d_new_vz
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Обмен указателей (swap), чтобы "новые" данные стали "старыми" для следующего шага
        float *tmp;
        tmp = d_x; d_x = d_new_x; d_new_x = tmp;
        tmp = d_y; d_y = d_new_y; d_new_y = tmp;
        tmp = d_z; d_z = d_new_z; d_new_z = tmp;
        tmp = d_vx; d_vx = d_new_vx; d_new_vx = tmp;
        tmp = d_vy; d_vy = d_new_vy; d_new_vy = tmp;
        tmp = d_vz; d_vz = d_new_vz; d_new_vz = tmp;

        current_time += DT;

        // Копируем обратно на хост для вывода
        // В реальных задачах вывод делают не каждый шаг, чтобы не тормозить GPU
        CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));

        printf("%.4f", current_time);
        for (int i = 0; i < n; i++) {
            printf(", %f, %f, %f", h_x[i], h_y[i], h_z[i]);
        }
        printf("\n");
    }

    // Освобождение памяти
    free(h_m); free(h_x); free(h_y); free(h_z); free(h_vx); free(h_vy); free(h_vz);
    cudaFree(d_m);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_new_x); cudaFree(d_new_y); cudaFree(d_new_z);
    cudaFree(d_new_vx); cudaFree(d_new_vy); cudaFree(d_new_vz);

    return 0;
}