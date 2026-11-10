#!/bin/bash
# Henrik's simcardemsx rodero_05 example (the port of em_tref7), reading physcardems' case.
#   T_END=20 OUT=runs/rodero_05/simcardemsx_smoke sbatch --qos=short --time=00:20:00 archer2/simcardemsx_rodero.slurm
#   sbatch archer2/simcardemsx_rodero.slurm         # full beat, into runs/rodero_05/simcardemsx_em_tref7
#SBATCH --job-name=simcardemsx_rodero
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=runs/_logs/simcardemsx_rodero_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
pce_container_env
SIMCARDEMSX_VENV="${SIMCARDEMSX_VENV:-$PCE_WORK/simcardemsx_venv}"
SIMCARDEMSX_SRC="${SIMCARDEMSX_SRC:-$PCE_WORK/simcardemsx}"
T_END="${T_END:-800}"
OUT="$(realpath -m "$PCE_REPO/${OUT:-runs/rodero_05/simcardemsx_em_tref7}")"
mkdir -p "$OUT"

cd "$SIMCARDEMSX_SRC/examples/rodero_05"
echo "$(date +%T) node $(hostname), simcardemsx $(git -C "$SIMCARDEMSX_SRC" rev-parse --short HEAD), t_end $T_END, out $OUT"
singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$SIMCARDEMSX_VENV/bin/python3" main.py \
  --case-dir "$PCE_REPO/cases/rodero_05" --t-end "$T_END" --output-dir "$OUT" --overwrite 2>&1
echo "$(date +%T) exit code $?"