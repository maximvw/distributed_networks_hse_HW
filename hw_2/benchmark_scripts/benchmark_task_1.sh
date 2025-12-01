#!/bin/bash

# Настройки
PROGRAM="./o_files/task1"
POINTS=10000000  # 10 миллионов точек (достаточно много, чтобы заметить разницу)
RUNS=7           # Количество запусков для усреднения
OUTPUT_FILE="results/benchmark_results.txt"

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
    # macOS
    MODEL=$(sysctl -n machdep.cpu.brand_string)
    CORES=$(sysctl -n hw.physicalcpu)
    LOGICAL=$(sysctl -n hw.logicalcpu)
else
    # Linux
    MODEL=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    CORES=$(grep -c ^processor /proc/cpuinfo) # Это логические ядра
    LOGICAL=$CORES
fi

echo "CPU: $MODEL" | tee -a $OUTPUT_FILE
echo "Физических ядер: $CORES" | tee -a $OUTPUT_FILE
echo "Логических потоков: $LOGICAL" | tee -a $OUTPUT_FILE
echo "Количество точек для теста: $POINTS" | tee -a $OUTPUT_FILE
echo "Количество запусков для усреднения: $RUNS" | tee -a $OUTPUT_FILE
echo "--------------------------------------" >> $OUTPUT_FILE

# Заголовок таблицы
printf "%-10s | %-15s | %-10s\n" "Потоки" "Среднее время (с)" "Ускорение" | tee -a $OUTPUT_FILE
echo "-----------|-----------------|-----------" | tee -a $OUTPUT_FILE

# Переменная для времени выполнения в 1 поток (для расчета ускорения)
TIME_1_THREAD=0

# 2. Цикл замеров
for threads in 1 2 4 8 12 16; do
    total_time=0
    
    # Запускаем N раз
    for (( i=1; i<=RUNS; i++ )); do
        # Запуск программы и перехват вывода. 
        # Мы ищем строку "Time taken: X.XXXX" и берем 3-е слово (число)
        output=$($PROGRAM $threads $POINTS)
        
        # Извлекаем время из вывода (парсинг строки Time taken: ...)
        run_time=$(echo "$output" | awk '/Time taken:/ {print $3}')
        
        # Складываем время (используем awk для float арифметики)
        total_time=$(awk "BEGIN {print $total_time + $run_time}")
    done

    # Считаем среднее
    avg_time=$(awk "BEGIN {print $total_time / $RUNS}")
    
    # Расчет ускорения (Speedup = T1 / Tn)
    if [ "$threads" -eq 1 ]; then
        TIME_1_THREAD=$avg_time
        speedup=1.00
    else
        speedup=$(awk "BEGIN {printf \"%.2f\", $TIME_1_THREAD / $avg_time}")
    fi

    # Красивый вывод в консоль и файл
    printf "%-10d | %-15.4f | %-10s\n" "$threads" "$avg_time" "$speedup" | tee -a $OUTPUT_FILE

done

echo "" >> $OUTPUT_FILE
echo "Готово! Результаты сохранены в $OUTPUT_FILE"