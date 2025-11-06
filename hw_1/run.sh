#!/bin/bash
# Универсальный скрипт для запуска всех MPI задач
# Примеры:
#   ./run.sh task1_pi 10000000
#   ./run.sh task2_matvec_block 1000
#   ./run.sh task4_dirichlet 1000 500

prog=$1
shift

if [ -z "$prog" ]; then
  echo "Usage: $0 <program_name> [args...]"
  exit 1
fi

mkdir -p results o_files

outfile="results/${prog}_results.csv"

# ---------- Заголовки ----------
case "$prog" in
  task1_pi)          header="pi_est,points,procs,time" ;;
  task2_matvec_rows) header="N,procs,time" ;;
  task2_matvec_cols) header="N,procs,time" ;;
  task2_matvec_block)header="N,procs,time,px,py" ;;
  task3_cannon)      header="N,procs,time" ;;
  task4_dirichlet)   header="N,procs,px,py,itmax,time" ;;
  *) echo "Unknown program name: $prog"; exit 1 ;;
esac

echo "$header" > "$outfile"

# ---------- Кол-во процессов ----------
if [[ "$prog" == "task2_matvec_block" || "$prog" == "task3_cannon" || "$prog" == "task4_dirichlet" ]]; then
  procs=(1 4)
else
  procs=(1 2 4 6)
fi

# ---------- Запуск ----------
echo "▶ Running $prog ..."
for p in "${procs[@]}"; do
  if [[ "$prog" == "task1_pi" ]]; then
    total_points=${1:-10000000}
    points_per_proc=$(( total_points / p ))
    echo "  $p processes (${points_per_proc} points/proc)"
    mpirun --oversubscribe -np "$p" "o_files/$prog" "$points_per_proc" 2>/dev/null \
      | grep -v "^#" | grep -v "$header" >> "$outfile"

  elif [[ "$prog" == "task2_matvec_rows" || "$prog" == "task2_matvec_cols" ]]; then
    N=${1:-1000}
    echo "  $p processes (N=$N)"
    mpirun --oversubscribe -np "$p" "o_files/$prog" "$N" 2>/dev/null \
      | grep -v "^#" | grep -v "$header" >> "$outfile"

  elif [[ "$prog" == "task2_matvec_block" ]]; then
    N=${1:-1000}
    echo "  $p processes (N=$N)"
    mpirun --oversubscribe -np "$p" "o_files/$prog" "$N" 2>/dev/null \
      | grep -v "^#" | grep -v "$header" >> "$outfile"

  elif [[ "$prog" == "task3_cannon" ]]; then
    N=${1:-512}
    echo "  $p processes (N=$N)"
    mpirun --oversubscribe -np "$p" "o_files/$prog" "$N" 2>/dev/null \
      | grep -v "^#" | grep -v "$header" >> "$outfile"

  elif [[ "$prog" == "task4_dirichlet" ]]; then
    N=${1:-1000}
    itmax=${2:-500}
    px=$(awk "BEGIN{printf \"%d\", sqrt($p)}")
    py=$px
    echo "  $p processes (${px}x${py}, N=$N, itmax=$itmax)"
    mpirun --oversubscribe -np "$p" "o_files/$prog" "$N" "$itmax" "$px" "$py" 2>/dev/null \
      | grep -v "^#" | grep -v "$header" >> "$outfile"
  fi
done

echo "Results saved to $outfile"
