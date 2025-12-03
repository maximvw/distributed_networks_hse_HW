#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Гравитационная постоянная
#define G 6.67430e-11
// Шаг по времени (Delta t). 
// В лабораторных работах его либо подбирают, либо задают жестко.
// Для точности возьмем 0.01 сек.
#define DT 0.01

typedef struct {
    double m;          // Масса
    double x, y, z;    // Координаты
    double vx, vy, vz; // Скорости
    double fx, fy, fz; // Силы
} Particle;

int main(int argc, char *argv[]) {
    // Проверка аргументов командной строки
    // Формат: ./program tend filename
    if (argc != 3) {
        printf("Usage: %s <tend> <filename>\n", argv[0]);
        return 1;
    }

    double tend = atof(argv[1]);
    char *filename = argv[2];

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

    // Выделение памяти под массив частиц
    Particle *particles = (Particle *)malloc(n * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(fp);
        return 1;
    }

    // Чтение данных из файла
    // В PDF написано "файл с массами", но в формате строки описано 6 чисел.
    // Обычно в N-body задачах масса необходима. 
    // Предполагаем формат: m x y z vx vy vz (7 чисел), так как без массы физика не работает.
    for (int i = 0; i < n; i++) {
        // Если в вашем файле 6 чисел (нет массы), измените fscanf и задайте p.m = 1.0 вручную.
        if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", 
            &particles[i].m, 
            &particles[i].x, &particles[i].y, &particles[i].z, 
            &particles[i].vx, &particles[i].vy, &particles[i].vz) != 7) {
            
            fprintf(stderr, "Error reading particle data at index %d. Expected 7 values (m x y z vx vy vz).\n", i);
            free(particles);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    // Основной цикл по времени
    int steps = (int)(tend / DT);
    
    // Вывод начального состояния (t=0)
    printf("0.000000");
    for (int i = 0; i < n; i++) {
        printf(",%f,%f,%f", particles[i].x, particles[i].y, particles[i].z);
    }
    printf("\n");

    for (int s = 1; s <= steps; s++) {
        double current_time = s * DT;

        // 1. Обнуление сил перед шагом
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }

        // 2. Расчет сил взаимодействия (Закон всемирного тяготения)
        // Используем 3-й закон Ньютона: F_ji = -F_ij
        // Внешний цикл распараллеливаем. Так как мы пишем в particles[j] (чужой поток),
        // нужны атомарные операции для корректности.
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                
                double dist_sq = dx*dx + dy*dy + dz*dz;
                // Добавляем малую величину eps, чтобы избежать деления на 0 при столкновении
                double dist = sqrt(dist_sq + 1e-10); 
                double dist_cube = dist * dist * dist;
                double f_mag;

                // if (dist == 0){
                    // f_mag = 0;
                // } else {
                f_mag = G * particles[i].m * particles[j].m / dist_cube;
                // }
                double fx = f_mag * dx;
                double fy = f_mag * dy;
                double fz = f_mag * dz;

                // Сила действует на i в сторону j (вектор r_j - r_i)
                #pragma omp atomic
                particles[i].fx += fx;
                #pragma omp atomic
                particles[i].fy += fy;
                #pragma omp atomic
                particles[i].fz += fz;

                // Противодействующая сила на j (3-й закон Ньютона)
                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;
                #pragma omp atomic
                particles[j].fz -= fz;
            }
        }

        // 3. Метод Эйлера: обновление скорости и координаты
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double ax = particles[i].fx / particles[i].m;
            double ay = particles[i].fy / particles[i].m;
            double az = particles[i].fz / particles[i].m;

            // Обновляем координаты: x(n) = x(n-1) + v(n-1)*dt
            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
            particles[i].z += particles[i].vz * DT;

            // Обновляем скорости: v(n) = v(n-1) + a(n-1)*dt
            particles[i].vx += ax * DT;
            particles[i].vy += ay * DT;
            particles[i].vz += az * DT;
        }

        // 4. Вывод текущего состояния в CSV
        printf("%f", current_time);
        for (int i = 0; i < n; i++) {
            printf(",%f,%f,%f", particles[i].x, particles[i].y, particles[i].z);
        }
        printf("\n");
    }

    free(particles);
    return 0;
}