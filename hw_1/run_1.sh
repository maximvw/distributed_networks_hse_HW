#!/bin/bash
# Пример запуска и сбора времени в CSV
# Использование:
#   bash /run_1.sh task1_pi 10000000

prog=$1
shift

if [ $# -lt 1 ]; then
  echo "Usage: $0 <program_name> <total_points> [other_args...]"
  exit 1
fi

total_points=$1
shift

mkdir -p results o_files

outfile="results/${prog}_results.csv"
echo "pi_est,points,procs,time" > "$outfile"


for p in 1 2 4 6; do

  points_per_proc=$(( total_points / p ))
  echo "Running $prog with $p processes (${points_per_proc} points per process)"
  mpirun -np "$p" "o_files/$prog" "$points_per_proc" "$@" 2>/dev/null | grep -v "^#" >> "$outfile"
done

echo "saved results to $outfile"
