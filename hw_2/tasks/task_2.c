#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Гравитационная постоянная
#define G 6.67430e-11
// Шаг по времени
#define DT 0.01

typedef struct {
    double m;          // Масса
    double x, y, z;    // Координаты
    double vx, vy, vz; // Скорости
    double fx, fy, fz; // Силы
} Particle;

int main(int argc, char *argv[]) {
    // Формат: ./program nthreads tend filename
    if (argc != 4) {
        printf("Usage: %s <nthreads> <tend> <filename>\n", argv[0]);
        return 1;
    }

    int nthreads = atoi(argv[1]);
    double tend = atof(argv[2]);
    char *filename = argv[3];

    // Установка количества потоков
    omp_set_num_threads(nthreads);

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

    Particle *particles = (Particle *)malloc(n * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(fp);
        return 1;
    }

    // Чтение данных.
    for (int i = 0; i < n; i++) {
        int read_count = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", 
            &particles[i].m, 
            &particles[i].x, &particles[i].y, &particles[i].z, 
            &particles[i].vx, &particles[i].vy, &particles[i].vz);
            
        if (read_count != 7) {
            fprintf(stderr, "Error reading particle %d. Expected 7 values (m x y z vx vy vz).\n", i);
            fprintf(stderr, "If your file has only 6 values per line, the PDF description implies mass is missing/fixed.\n");
            free(particles);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    double start_time = omp_get_wtime();

    int steps = (int)(tend / DT);
    
    // Вывод начального состояния (t=0)
    // Формат CSV: t, x1, y1, x2, y2... (в PDF только x, y, но код считает z, оставим z для точности)
    printf("0.000000");
    for (int i = 0; i < n; i++) {
        printf(",%f,%f,%f", particles[i].x, particles[i].y, particles[i].z);
    }
    printf("\n");

    // Основной цикл
    for (int s = 1; s <= steps; s++) {
        double current_time = s * DT;

        // 1. Обнуление сил
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }

        // 2. Расчет сил (с учетом 3-го закона Ньютона)
        // schedule(dynamic) помогает, так как нагрузка во внутреннем цикле уменьшается с ростом i
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                
                double dist_sq = dx*dx + dy*dy + dz*dz;
                // Защита от деления на ноль + softening
                double dist = sqrt(dist_sq + 1e-10); 
                double dist_cube = dist * dist * dist;
                
                double f_mag = G * particles[i].m * particles[j].m / dist_cube;
                
                double fx = f_mag * dx;
                double fy = f_mag * dy;
                double fz = f_mag * dz;

                // Обновляем i (нужен atomic, т.к. другие потоки могут обновлять i как j)
                #pragma omp atomic
                particles[i].fx += fx;
                #pragma omp atomic
                particles[i].fy += fy;
                #pragma omp atomic
                particles[i].fz += fz;

                // Обновляем j (нужен atomic, т.к. j - чужой индекс)
                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;
                #pragma omp atomic
                particles[j].fz -= fz;
            }
        }

        // 3. Интеграция по Эйлеру
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double ax = particles[i].fx / particles[i].m;
            double ay = particles[i].fy / particles[i].m;
            double az = particles[i].fz / particles[i].m;

            particles[i].vx += ax * DT;
            particles[i].vy += ay * DT;
            particles[i].vz += az * DT;

            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
            particles[i].z += particles[i].vz * DT;

        }

        // 4. Вывод
        printf("%f", current_time);
        for (int i = 0; i < n; i++) {
            printf(",%f,%f,%f", particles[i].x, particles[i].y, particles[i].z);
        }
        printf("\n");
    }
    double end_time = omp_get_wtime();
    fprintf(stderr, "Time taken: %.4f seconds\n", end_time - start_time);

    free(particles);
    return 0;
}
