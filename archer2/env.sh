# Shared ARCHER2 settings, sourced by the scripts in this folder.
# Override any of these by exporting them before running, e.g. PCE_WORK=/work/<proj>/<proj>/<user>
PCE_WORK="${PCE_WORK:-/mnt/lustre/a2fs-work4/work/e952/e952/jenang}"
PCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PCE_IMAGE="${PCE_IMAGE:-$PCE_WORK/containers/dolfinx_v0.10.0.sif}"
PCE_VENV="${PCE_VENV:-$PCE_WORK/physcardems_venv}"   # layered on the container's own Python environment
PCE_CACHE="${PCE_CACHE:-$PCE_WORK/.cache}"           # pip, singularity and the FEniCSx JIT cache

# Environment for runs inside the container
pce_container_env() {
  export SINGULARITYENV_XDG_CACHE_HOME="$PCE_CACHE"
  export SINGULARITYENV_MPLCONFIGDIR="$PCE_CACHE/matplotlib"
  export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1
  # serial runs: keep the container's MPI from looking for Slurm's process manager
  for v in $(env | grep -oE '^(PMI|PMIX|SLURM_STEP|SLURM_PROCID)[A-Z_]*'); do unset "$v"; done
}