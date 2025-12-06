### run task_1:
1) compile: gcc-15 -fopenmp -o o_files/task_1 tasks/task_1/task_1.c -lm
2) run: o_files/task_1 4 1000000
3) benchmarking: bash ./benchmark_scripts/benchmark_task_1.sh
4) check graphic: uv run ./visualizers/task_1.py   

### run task_2 openmp:
1) compile: gcc-15 -fopenmp -o o_files/task_2 tasks/task_2/task_2.c -lm

2) run: 
    atom: ./o_files/task_2 4 10.0 data/part_2/atom.txt > ./results/task_results/atom.csv
    2stars: ./o_files/task_2 4 10.0 data/part_2/2stars.txt > ./results/task_results/2stars.csv
    sputnik: ./o_files/task_2 4 10.0 data/part_2/sputnik.txt > ./results/task_results/sputnik.csv
    sun_and_earth: ./o_files/task_2 4 10.0 data/part_2/sun_and_earth.txt > ./results/task_results/sun_and_earth.csv
    star_and_2planets: ./o_files/task_2 4 10.0 data/part_2/star_and_2planets.txt > ./results/task_results/star_and_2planets.csv

3) visualize: 
    atom: uv run visualizers/task_2.py ./results/task_results/atom.csv
    2stars: uv run visualizers/task_2.py ./results/task_results/2stars.csv
    sputnik: uv run visualizers/task_2.py ./results/task_results/sputnik.csv
    sun_and_earth: uv run visualizers/task_2.py ./results/task_results/sun_and_earth.csv
    star_and_2planets: uv run visualizers/task_2.py ./results/task_results/star_and_2planets.csv

4) benchmarking: bash ./benchmark_scripts/benchmark_task_2.sh


### run task_2_cuda:
1) compile: nvcc -arch=sm_75 -O3 -o o_files/task_2_cuda tasks/task_2/task_2_cuda.cu

2) run: 
    atom: ./o_files/task_2_cuda 4 10.0 data/part_2/atom.txt > ./results/task_results/atom_cuda.csv
    2stars: ./o_files/task_2_cuda 4 10.0 data/part_2/2stars.txt > ./results/task_results/2stars_cuda.csv
    sputnik: ./o_files/task_2_cuda 4 10.0 data/part_2/sputnik.txt > ./results/task_results/sputnik_cuda.csv
    sun_and_earth: ./o_files/task_2_cuda 4 10.0 data/part_2/sun_and_earth.txt > ./results/task_results/sun_and_earth_cuda.csv
    star_and_2planets: ./o_files/task_2_cuda 4 10.0 data/part_2/star_and_2planets.txt > ./results/task_results/star_and_2planets_cuda.csv

3) visualize: 
    atom: uv run visualizers/task_2.py ./results/task_results/atom_cuda.csv
    2stars: uv run visualizers/task_2.py ./results/task_results/2stars_cuda.csv
    sputnik: uv run visualizers/task_2.py ./results/task_results/sputnik_cuda.csv
    sun_and_earth: uv run visualizers/task_2.py ./results/task_results/sun_and_earth_cuda.csv
    star_and_2planets: uv run visualizers/task_2.py ./results/task_results/star_and_2planets_cuda.csv

4) benchmarking: bash ./benchmark_scripts/benchmark_task_2_cuda.sh


### run task_3:
1) compile (lock from library): gcc-15 -o o_files/task_3 tasks/task_3/pth_ll_rwl.c tasks/task_3/my_rand.c -lm
1) compile (custom lock): gcc-15 -o o_files/task_3_custom tasks/task_3/pth_ll_rwl_custom.c tasks/task_3/my_rand.c -lm

2) run library: /o_files/task_3 8 
2) run custom: /o_files/task_3_custom 8 

3) benchmarking: bash ./benchmark_scripts/benchmark_task_3.sh
