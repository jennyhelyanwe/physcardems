#!/bin/bash
# Run many single-core simulations side by side on one ARCHER2 node.
#   cd <repo root> && sbatch archer2/sweep.slurm runs/<study>/configs.txt
#   short test:  NPAR=8 sbatch --qos=short --time=00:20:00 archer2/sweep.slurm runs/ta_lhs_test/configs.txt
#   other run script and threads per run (default run_mech_ta.py, 1 thread):
#     PCE_SCRIPT=scripts/run_em_lvonly.py PCE_THREADS=8 NPAR=3 sbatch archer2/sweep.slurm <list>
# configs.txt: one config path per line (relative to the repo root, or absolute).
# For more than one node, split the list and submit one job per part.
#SBATCH --job-name=pce_sweep
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=runs/_logs/sweep_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
LIST="$1"
NPAR=${NPAR:-64}        # concurrent runs on the node: about 256 GB / (memory per run)
cd "$PCE_REPO"
pce_container_env
# threads per run inside the container (env.sh sets 1); NPAR x PCE_THREADS should stay <= 128
T=${PCE_THREADS:-1}
export SINGULARITYENV_OMP_NUM_THREADS=$T SINGULARITYENV_OPENBLAS_NUM_THREADS=$T SINGULARITYENV_MKL_NUM_THREADS=$T
export PCE_SCRIPT="${PCE_SCRIPT:-scripts/run_mech_ta.py}"
export PCE_RUN="singularity exec -B $PCE_WORK $PCE_IMAGE $PCE_VENV/bin/python3"
exec bash scripts/run_pool.sh "$LIST" "$NPAR"