#!/bin/bash
# Short ARCHER2 test of run_em_lvonly.py (20 min QoS): one LV ellipsoid case from t = 0 to 200 ms,
# i.e. through preload and into isovolumic contraction.
#   cd <repo root> && sbatch archer2/test_lvonly_short.slurm [config]
# Default config: configs/lv_ellipsoid/baseline.toml (6 mm student case).
#SBATCH --job-name=pce_lv_test
#SBATCH --account=e952
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=runs/_logs/test_lvonly_short_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/archer2/env.sh"
cd "$PCE_REPO"
pce_container_env

SRC=${1:-configs/lv_ellipsoid/baseline.toml}
NAME=$(basename "$SRC" .toml)
mkdir -p runs/lv_ellipsoid
CFG=runs/lv_ellipsoid/test_short_${NAME}.toml
{ echo "outdir = \"runs/lv_ellipsoid/test_short_${NAME}\""
  sed -e 's/^t_end_ms = .*/t_end_ms = 200.0/' "$SRC"
} > "$CFG"

echo "$(date +%T) node $(hostname), config $CFG (from $SRC)"
singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$PCE_VENV/bin/python3" scripts/run_em_lvonly.py "$CFG" 2>&1
echo "$(date +%T) exit code $?"