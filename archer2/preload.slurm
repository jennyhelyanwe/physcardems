#!/bin/bash
# Shared preload for prescribed-Ta studies: runs base.toml to 130 ms (just before the first
# activation, so Ta is still zero) and checkpoints every 10 ms. Every LHS sample restarts
# from here, so the preload is computed once.
#   cd <repo root> && sbatch archer2/preload.slurm
#SBATCH --job-name=pce_preload
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=runs/_logs/preload_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
cd "$PCE_REPO"
pce_container_env

CFG=configs/ta_sweep/preload.toml
{ echo 'outdir = "runs/rodero_05/preload"'
  sed -e 's/^t_end_ms = .*/t_end_ms = 130.0/' \
      -e 's/^checkpoint_every_ms = .*/checkpoint_every_ms = 10.0/' configs/ta_sweep/base.toml
} > "$CFG"

echo "$(date +%T) node $(hostname), config $CFG"
singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$PCE_VENV/bin/python3" scripts/run_mech_ta.py "$CFG" 2>&1
echo "$(date +%T) exit code $?"
ls runs/rodero_05/preload/checkpoints/