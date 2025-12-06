#!/bin/bash

# Настройки
PROGRAM="./o_files/task_1"
POINTS=10000000  # 10 миллионов точек
RUNS=10          # Количество запусков для усреднения
OUTPUT_FILE="results/benchmarks/benchmark_1.txt"

# Проверка наличия программы
if [ ! -f "$PROGRAM" ]; then
    echo "Ошибка: Программа $PROGRAM не найдена. Скомпилируйте её сначала."
    exit 1
fi

# Очистка файла результатов
echo "Отчет о производительности" > $OUTPUT_FILE
echo "--------------------------------------" >> $OUTPUT_FILE
date >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 1. Информация об аппаратной архитектуре
echo "=== Аппаратная конфигурация ===" | tee -a $OUTPUT_FILE
if [[ "$OSTYPE" == "darwin"* ]]; then
    MODEL=$(sysctl -n machdep.cpu.brand_string)
    CORES=$(sysctl -n hw.physicalcpu)
    LOGICAL=$(sysctl -n hw.logicalcpu)
else
    MODEL=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    CORES=$(grep -c ^processor /proc/cpuinfo) # Обычно это логические ядра в Linux
    LOGICAL=$CORES
fi

echo "CPU: $MODEL" | tee -a $OUTPUT_FILE
echo "Физических ядер: $CORES" | tee -a $OUTPUT_FILE
echo "Логических потоков: $LOGICAL" | tee -a $OUTPUT_FILE
echo "Количество точек для теста: $POINTS" | tee -a $OUTPUT_FILE
echo "Количество запусков для усреднения: $RUNS" | tee -a $OUTPUT_FILE
echo "--------------------------------------" >> $OUTPUT_FILE

# Заголовок таблицы (Добавлена колонка "Эффективность")
printf "%-10s | %-15s | %-10s | %-15s\n" "Потоки" "Среднее время (с)" "Ускорение" "Эффективность" | tee -a $OUTPUT_FILE
echo "-----------|-----------------|------------|----------------" | tee -a $OUTPUT_FILE

# Переменная для времени выполнения в 1 поток
TIME_1_THREAD=0

# 2. Цикл замеров
for threads in 1 2 4 8 12 16; do
    total_time=0
    
    # Запускаем N раз
    for (( i=1; i<=RUNS; i++ )); do
        # Запуск программы. Ожидается вывод вида "Time taken: X.XXXX"
        output=$($PROGRAM $threads $POINTS)
        
        # Извлекаем время из вывода
        run_time=$(echo "$output" | awk '/Time taken:/ {print $3}')
        
        # Складываем время
        total_time=$(awk "BEGIN {print $total_time + $run_time}")
    done

    # Считаем среднее
    avg_time=$(awk "BEGIN {print $total_time / $RUNS}")
    
    # Расчет ускорения (Speedup) и Эффективности (Efficiency)
    if [ "$threads" -eq 1 ]; then
        TIME_1_THREAD=$avg_time
        speedup=1.00
        efficiency=1.00
    else
        # Ускорение = T1 / Tn
        speedup=$(awk "BEGIN {printf \"%.2f\", $TIME_1_THREAD / $avg_time}")
        
        # Эффективность = Speedup / Threads
        # Примечание: speedup уже отформатирован как строка, awk это поймет
        efficiency=$(awk "BEGIN {printf \"%.2f\", $speedup / $threads}")
    fi

    # Вывод результатов в консоль и файл (Добавлена эффективность)
    printf "%-10d | %-15.4f | %-10s | %-15s\n" "$threads" "$avg_time" "$speedup" "$efficiency" | tee -a $OUTPUT_FILE

done

echo "" >> $OUTPUT_FILE
echo "Готово! Результаты сохранены в $OUTPUT_FILE"