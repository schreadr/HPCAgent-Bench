#!/bin/bash
#SBATCH --job-name=bench_sweep
#SBATCH -A dphpc
#SBATCH --gpus=5060ti:1          # or gb10:1 on GB10
#SBATCH --time=02:00:00          # billed for the WHOLE allocation, so size it to the sweep
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

# Submit from the HPCAgent-Bench checkout:  sbatch run_sweep.sbatch
cd "$SLURM_SUBMIT_DIR" || exit 1

# ---- edit these -------------------------------------------------------------------------------
KERNELS=(
    subset_sum nqueens qe_addusxx_g comet_int4_gemm
    distribution_search eigh_generalized householder_qr largest_eigenval
    cloudsc_lu_solver mixed_precision_ir qe_newdxx_g portfolio_optimization
)
DTYPE=float32          # a kernel whose manifest lists only fp64 needs float64
declare -A DTYPE_FOR=(
    mixed_precision_ir=float64
)
REPS=3                 # repetitions for the preset M run
PER_RUN_LIMIT=20m      # a hung kernel must not burn the whole allocation
RUN="uv run hpcagent-bench"
# Container variant for final numbers (swap the line above):
# RUN="apptainer exec --nv $HOME/judge-nvidia-x86_64.sif hpcagent-bench"
# If you use the separate x86 environment from the how-to, export it before sbatch or here:
# export UV_PROJECT_ENVIRONMENT=.venv-x86
# -----------------------------------------------------------------------------------------------

# Environment the harness checks before it starts (must be set in this shell, before any run)
ulimit -s "$(ulimit -H -s)"
export OMP_STACKSIZE=512M
export OMP_THREAD_LIMIT=$(nproc)
export HPCAGENT_BENCH_GRADING_SEAL=false

set -o pipefail
LOGDIR="logs/sweep_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"
FAILED=()

run() {  # run <label> <log name> <command...>
    local label="$1" log="$LOGDIR/$2.log"
    shift 2
    echo "=== $label  $(date +%T) ==="
    if ! timeout "$PER_RUN_LIMIT" "$@" 2>&1 | tee "$log"; then
        echo "!!! FAILED or timed out: $label"
        FAILED+=("$label")
    fi
}

for k in "${KERNELS[@]}"; do
    # preset M
    run "$k  M" "${k}_M" \
        $RUN run-framework -f triton -b "$k" -p M -d "$DTYPE" -r "$REPS"
    # default fuzzed preset (official evaluation; no -p)
    run "$k  fuzzed" "${k}_fuzzed" \
        $RUN run-framework -f triton -b "$k" -d "$DTYPE"
done

# Median runtimes for this sweep's kernels, from the results database (also saved as CSV).
# Plain python3 is enough here: sqlite3 is in the standard library.
echo
echo "=== median runtimes ==="
python3 - "$LOGDIR/medians.csv" "${KERNELS[@]}" <<'PYEOF'
import csv, sqlite3, sys
out, kernels = sys.argv[1], sys.argv[2:]
conn = sqlite3.connect("results/hpcagent_bench0.db")
q = ("SELECT column, kernel, preset, datatype, median_ms, validated FROM canon "
     "WHERE kernel IN (%s) ORDER BY kernel, preset, column" % ",".join("?" * len(kernels)))
rows = conn.execute(q, kernels).fetchall()
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["framework", "kernel", "preset", "datatype", "median_ms", "validated"])
    w.writerows(rows)
print(f"{'framework':<12} {'kernel':<24} {'preset':<8} {'datatype':<10} {'median_ms':>12} {'validated':>10}")
print("-" * 80)
for col, kern, pre, dt, med, val in rows:
    m = f"{med:.3f} ms" if med is not None else "N/A"
    print(f"{str(col):<12} {str(kern):<24} {str(pre):<8} {str(dt):<10} {m:>12} {str(val):>10}")
print("saved to", out)
PYEOF
# echo
# echo "=== validation / run summary ==="
# $RUN run-framework --summarize results/hpcagent_bench0.db

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed or timed out (${#FAILED[@]}):"
    printf '  %s\n' "${FAILED[@]}"
else
    echo "All runs finished without a crash or timeout (check validation in the summary above)."
fi
