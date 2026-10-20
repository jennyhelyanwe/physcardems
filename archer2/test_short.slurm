#!/bin/bash
# Single-core test of run_mech_ta.py on the short QoS (20 min limit), from t = 0.
#   cd <repo root> && sbatch archer2/test_short.slurm
#SBATCH --job-name=pce_test
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=runs/_logs/test_short_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
cd "$PCE_REPO"
pce_container_env

CFG=configs/ta_sweep/test_short.toml
{ echo 'outdir = "runs/rodero_05/test_short"'
  sed -e 's/^t_end_ms = .*/t_end_ms = 200.0/' configs/ta_sweep/base.toml
} > "$CFG"

echo "$(date +%T) node $(hostname), config $CFG"
singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$PCE_VENV/bin/python3" scripts/run_mech_ta.py "$CFG" 2>&1
echo "$(date +%T) exit code $?"