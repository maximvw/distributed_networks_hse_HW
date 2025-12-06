#!/bin/bash

# Настройки
PROGRAM="./o_files/task_2_cuda"
TEND=1.0
RUNS=10        # Количество запусков для усреднения
OUTPUT_FILE="results/benchmarks/benchmark_2_cuda.txt"

# Проверка наличия программы
if [ ! -f "$PROGRAM" ]; then
    echo "Ошибка: Программа $PROGRAM не найдена. Скомпилируйте её сначала."
    exit 1
fi

# Очистка файла результатов
echo "Отчет о производительности (CUDA)" > $OUTPUT_FILE
echo "--------------------------------------" >> $OUTPUT_FILE
date >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 1. Информация об аппаратной архитектуре
echo "=== Аппаратная конфигурация ===" | tee -a $OUTPUT_FILE

# Информация о CPU (для контекста хоста)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    MODEL=$(sysctl -n machdep.cpu.brand_string)
else
    # Linux
    MODEL=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
fi
echo "Host CPU: $MODEL" | tee -a $OUTPUT_FILE

# Информация о GPU
if command -v nvidia-smi &> /dev/null; then
    # Получаем имя первой видеокарты
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    # Получаем версию драйвера и CUDA
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo "GPU Device: $GPU_MODEL" | tee -a $OUTPUT_FILE
    echo "GPU Driver: $DRIVER_VER" | tee -a $OUTPUT_FILE
else
    echo "GPU: Информация недоступна (nvidia-smi не найден)" | tee -a $OUTPUT_FILE
fi

echo "T_end теста: $TEND" | tee -a $OUTPUT_FILE
echo "Количество запусков для усреднения: $RUNS" | tee -a $OUTPUT_FILE
echo "--------------------------------------" >> $OUTPUT_FILE

# Заголовок таблицы
# Вместо "Потоки" теперь "Block Size" (размер блока)
printf "%-12s | %-15s | %-10s\n" "Block Size" "Среднее время (с)" "Ускорение" | tee -a $OUTPUT_FILE
echo "-------------|-----------------|-----------" | tee -a $OUTPUT_FILE

# Переменная для времени выполнения базового случая (первого в списке)
BASE_TIME=0
IS_FIRST_RUN=1

# 2. Цикл замеров
# Для CUDA используем степени двойки, начиная с 32 (размер варпа) до 1024 (макс. размер блока)
for block_size in 32 64 128 256 512 1024; do
    total_time=0
    
    # Запускаем N раз
    for (( i=1; i<=RUNS; i++ )); do
        # Запуск программы: ./program <threads_per_block> <tend> <file>
        output=$($PROGRAM $block_size $TEND data/part_2/atom_big.txt 2>&1 >/dev/null)
        
        # Извлекаем время
        run_time=$(echo "$output" | awk '/Time taken:/ {print $3}')
        
        # Проверка
        if [ -z "$run_time" ]; then
             echo "Ошибка: не удалось получить время выполнения. Проверьте вывод программы."
             echo "Вывод программы: $output"
             exit 1
        fi

        # Складываем время
        total_time=$(awk "BEGIN {print $total_time + $run_time}")
    done

    # Считаем среднее
    avg_time=$(awk "BEGIN {print $total_time / $RUNS}")
    
    # Расчет "ускорения" (относительно размера блока 32)
    # Здесь Speedup показывает, насколько удачный выбор размера блока влияет на скорость
    if [ "$IS_FIRST_RUN" -eq 1 ]; then
        BASE_TIME=$avg_time
        speedup=1.00
        IS_FIRST_RUN=0
    else
        speedup=$(awk "BEGIN {printf \"%.2f\", $BASE_TIME / $avg_time}")
    fi

    # Красивый вывод в консоль и файл
    printf "%-12d | %-15.4f | %-10s\n" "$block_size" "$avg_time" "$speedup" | tee -a $OUTPUT_FILE
done

echo "" >> $OUTPUT_FILE
echo "Готово! Результаты сохранены в $OUTPUT_FILE"