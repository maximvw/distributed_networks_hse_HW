#!/bin/bash

# ==========================================
# НАСТРОЙКИ ТЕСТА
# ==========================================
PROG_STD="./o_files/task_3"        # Библиотечная реализация
PROG_CUSTOM="./o_files/task_3_custom" # Ваша реализация
OUTPUT_FILE="results/benchmarks/benchmark_3_compare.txt"
RUNS=10  # Количество прогонов для усреднения

# Параметры ввода для программы (увеличил для наглядности нагрузки)
# Если будет работать слишком долго, уменьшите TOTAL_OPS
KEYS_IN_MAIN=10000      # Сколько ключей вставить до запуска потоков
TOTAL_OPS=200000        # Общее количество операций
SEARCH_PERCENT=0.8      # 80% поиска (читатели)
INSERT_PERCENT=0.1      # 10% вставок (писатели)
# Оставшиеся 0.1 (10%) - удаления (писатели)

# Создаем папку для результатов, если нет
mkdir -p $(dirname "$OUTPUT_FILE")

# Проверка наличия файлов
if [ ! -f "$PROG_STD" ] || [ ! -f "$PROG_CUSTOM" ]; then
    echo "Ошибка: Не найдены исполняемые файлы."
    echo "Убедитесь, что существуют $PROG_STD и $PROG_CUSTOM"
    exit 1
fi

# ==========================================
# ФУНКЦИЯ БЕНЧМАРКА
# ==========================================
run_benchmark() {
    local title=$1
    local prog_path=$2
    local time_1_thread=0

    echo "" | tee -a $OUTPUT_FILE
    echo "=== Тестирование: $title ===" | tee -a $OUTPUT_FILE
    printf "%-10s | %-15s | %-10s | %-15s\n" "Потоки" "Среднее время (с)" "Ускорение" "Эффективность" | tee -a $OUTPUT_FILE
    echo "-----------|-----------------|------------|----------------" | tee -a $OUTPUT_FILE

    # Перебираем количество потоков (добавил 12 и 16, если это M1/M2/M3 Pro/Max)
    for threads in 1 2 4 8 12 16; do
        total_time=0
        
        for (( i=1; i<=RUNS; i++ )); do
            # Формируем ввод для программы:
            # 1 строка: ключи, 2: опции, 3: % поиска, 4: % вставок
            input_str=$(printf "%s\n%s\n%s\n%s\n" "$KEYS_IN_MAIN" "$TOTAL_OPS" "$SEARCH_PERCENT" "$INSERT_PERCENT")
            
            # Запускаем, подаем input, ловим вывод
            output=$(echo "$input_str" | $prog_path $threads)
            
            # Парсим время: ищем строку "Elapsed time =" и берем 4-е слово
            run_time=$(echo "$output" | awk '/Elapsed time/ {print $4}')
            
            # Суммируем (awk умеет работать с научной нотацией e-04)
            total_time=$(awk "BEGIN {print $total_time + $run_time}")
        done

        # Среднее время
        avg_time=$(awk "BEGIN {print $total_time / $RUNS}")
        
        # Расчет метрик
        if [ "$threads" -eq 1 ]; then
            time_1_thread=$avg_time
            speedup=1.00
            efficiency=1.00
        else
            if (( $(echo "$avg_time > 0" | bc -l) )); then
                speedup=$(awk "BEGIN {printf \"%.2f\", $time_1_thread / $avg_time}")
                efficiency=$(awk "BEGIN {printf \"%.2f\", $speedup / $threads}")
            else
                speedup="N/A"
                efficiency="N/A"
            fi
        fi

        printf "%-10d | %-15.4f | %-10s | %-15s\n" "$threads" "$avg_time" "$speedup" "$efficiency" | tee -a $OUTPUT_FILE
    done
}

# ==========================================
# ОСНОВНОЕ ТЕЛО СКРИПТА
# ==========================================

# Очистка и заголовок файла
echo "Сравнение производительности RWLock (Library vs Custom)" > $OUTPUT_FILE
echo "Дата: $(date)" >> $OUTPUT_FILE
echo "-------------------------------------------------------" >> $OUTPUT_FILE

# Информация о системе (macOS/Linux)
echo "=== Аппаратная конфигурация ===" | tee -a $OUTPUT_FILE
if [[ "$OSTYPE" == "darwin"* ]]; then
    MODEL=$(sysctl -n machdep.cpu.brand_string)
    CORES=$(sysctl -n hw.physicalcpu)
    LOGICAL=$(sysctl -n hw.logicalcpu)
else
    MODEL=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    CORES=$(grep -c ^processor /proc/cpuinfo)
    LOGICAL=$CORES
fi

echo "CPU: $MODEL" | tee -a $OUTPUT_FILE
echo "Физических ядер: $CORES" | tee -a $OUTPUT_FILE
echo "Логических потоков: $LOGICAL" | tee -a $OUTPUT_FILE
echo "-------------------------------------------------------" >> $OUTPUT_FILE
echo "Параметры теста:" | tee -a $OUTPUT_FILE
echo "  Keys in Main: $KEYS_IN_MAIN" | tee -a $OUTPUT_FILE
echo "  Total Ops:    $TOTAL_OPS" | tee -a $OUTPUT_FILE
echo "  Search %:     $SEARCH_PERCENT" | tee -a $OUTPUT_FILE
echo "  Insert %:     $INSERT_PERCENT" | tee -a $OUTPUT_FILE
echo "-------------------------------------------------------" >> $OUTPUT_FILE

# Запуск тестов
run_benchmark "Standard pthread_rwlock" "$PROG_STD"
run_benchmark "Custom RWLock Implementation" "$PROG_CUSTOM"

echo ""
echo "Готово! Результаты сохранены в $OUTPUT_FILE"