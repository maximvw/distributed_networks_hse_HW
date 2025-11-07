prog=$1
shift

if [ -z "$prog" ]; then
  echo "Usage: $0 <program_name> [args...]"
  exit 1
fi

mkdir -p results o_files

outfile="results/${prog}_results.csv"

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

procs=(1 2 3 4 6)

echo "▶ Running $prog ..."
for p in "${procs[@]}"; do
  echo "  → $p processes"
  mpirun --oversubscribe -np "$p" "o_files/$prog" "$@" 2>/dev/null \
    | grep -v "^#" | grep -v "$header" >> "$outfile"
done

echo "Results saved to $outfile"
