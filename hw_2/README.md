### run task_1:
1) compile: gcc-15 -fopenmp -o o_files/task_1 tasks/task_1.c -lm
2) run: o_files/task_1 4 1000000
3) benchmarking: ./benchmark_scripts/benchmark_task_1.sh
4) check graphic: uv run ./visualizers/task_1.py   