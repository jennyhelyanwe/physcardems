#!/bin/bash
# physcardems' coupled em_tref7 run on rodero_05: the reference for the simcardemsx comparison.
#   sbatch archer2/physcardems_em_tref7.slurm        # writes runs/rodero_05/em_tref7
#SBATCH --job-name=physcardems_em_tref7
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=runs/_logs/physcardems_em_tref7_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
cd "$PCE_REPO"
pce_container_env
echo "$(date +%T) node $(hostname), physcardems $(git rev-parse --short HEAD)"
singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$PCE_VENV/bin/python3" \
  scripts/run_em.py configs/elife/em_tref7.toml 2>&1
echo "$(date +%T) exit code $?"