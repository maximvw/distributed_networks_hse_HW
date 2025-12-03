### run task_1:
1) compile: gcc-15 -fopenmp -o o_files/task_1 tasks/task_1.c -lm
2) run: o_files/task_1 4 1000000
3) benchmarking: ./benchmark_scripts/benchmark_task_1.sh
4) check graphic: uv run ./visualizers/task_1.py   

### run task_2 openmp:
1) compile: gcc-15 -fopenmp -o o_files/task_2 tasks/task_2.c -lm
2) run: ./o_files/task_2 10.0 data/part_2/input.txt > ./results/task_results/n_body.csv
3) visualize: uv run visualizers/task_2.py ./results/task_results/n_body.csv

### run task_2 cuda:
1) compile: nvcc -O3 -o o_files/task_2_cuda tasks/task_2.cu
2) run: ./o_files/task_2_cuda 10.0 data/part_2/input.txt > ./results/task_results/n_body_cuda.csv
3) visualize: uv run visualizers/task_2.py ./results/task_results/n_body_cuda.csv