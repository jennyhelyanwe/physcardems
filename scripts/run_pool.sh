#!/usr/bin/env bash
# Run a list of configs through run_mech_ta.py, NPAR at a time, one core each.
#   bash scripts/run_pool.sh runs/<study>/configs.txt [NPAR]
# Run from the repo root. Used directly inside the Docker container, and by archer2/sweep.slurm,
# which sets PCE_RUN to the container command. Logs go to runs/_logs/<config path>.log.
(
set -uo pipefail
LIST="$1"
NPAR="${2:-4}"
export PCE_RUN="${PCE_RUN:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/_logs

logname() { echo "runs/_logs/$(echo "$1" | tr '/' '_' | sed 's/\.\(toml\|json\)$//').log"; }
run_one() {
  $PCE_RUN scripts/run_mech_ta.py "$1" > "$(logname "$1")" 2>&1
  echo "$(date +%T) finished (exit $?) $1"
}
export -f run_one logname

mapfile -t CONFIGS < <(grep -v '^\s*$' "$LIST")
echo "$(date +%T) $(hostname): ${#CONFIGS[@]} configs, $NPAR at a time"

# Warm-up: let the first run compile the forms into the JIT cache before the rest start,
# so several processes don't compile the same forms at once.
run_one "${CONFIGS[0]}" &
first=$(logname "${CONFIGS[0]}")
until grep -q "Mechanics problem ready" "$first" 2>/dev/null || ! jobs %1 >/dev/null 2>&1; do sleep 5; done
echo "$(date +%T) forms compiled; starting the rest"

printf '%s\n' "${CONFIGS[@]:1}" | xargs -P $(( NPAR > 1 ? NPAR - 1 : 1 )) -I{} bash -c 'run_one "$@"' _ {}
wait
echo "$(date +%T) all done"
)